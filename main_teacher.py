# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import argparse
import datetime
import os
import socket
import subprocess
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter

from config import get_config
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler

# from models import build_model
from models.teacher import build_simmim
from optimizer import build_optimizer
from utils import (
    _find_free_port,
    auto_resume_helper,
    get_grad_norm,
    load_checkpoint,
    save_checkpoint,
)

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser("SimMIM pre-training script", add_help=False)
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--pretrained", type=str, help="path to pre-trained model")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for similarity loss")
    parser.add_argument("--launcher", type=str, default="slurm", help="The job launcer")
    parser.add_argument("--master_port", type=str, default="12365", help="The master port")

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    data_loader_train = build_loader(config, logger, is_pretrain=True)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_simmim(config, logger)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(
        model, broadcast_buffers=False, find_unused_parameters=True
    )
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(config, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img, mask, _) in enumerate(data_loader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        loss = model(img, mask)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def init_dist_slurm(backend, port=None, init_backend="torch", **kwargs) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    # Not sure when this environment variable could be None, so use a fallback
    local_rank_env = os.environ.get("SLURM_LOCALID", None)
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ["MASTER_PORT"] = "29500"
    # use MASTER_ADDR in the environment variable if it already exists
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(proc_id)

    torch.cuda.set_device(local_rank)

    if init_backend == "torch":
        dist.init_process_group(backend=backend, **kwargs)
    else:
        raise ValueError(
            'supported "init_backend" is "torch" or "deepspeed", ' f"but got {init_backend}"
        )
    dist.barrier()


def _find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


if __name__ == "__main__":
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    # currently supported launchers: slurm
    init_dist_slurm(backend="nccl", port=config.MASTER_PORT)

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
    )

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)

from .data_finetune import build_loader_finetune, build_test_loader
from .data_simmim import build_loader_simmim


def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_simmim(config, logger)
    else:
        return build_loader_finetune(config, logger)

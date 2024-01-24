#!/usr/bin/env bash

set -x

PARTITION="s2_bigdata"
# PARTITION="test_s2"

# train config
# NODE=2
# GPUS=16
# GPUS_PER_NODE=$((GPUS/NODE))
# CPUS_PER_TASK=8

# finetune config
NODE=1
GPUS=4
GPUS_PER_NODE=$((GPUS/NODE))
CPUS_PER_TASK=8

# using specific nodes
SRUN_ARGS=""
# SRUN_ARGS="--debug"
# SRUN_ARGS="${SRUN_ARGS} --nodelist=SH-IDC1-10-140-24-128,"
# SRUN_ARGS="${SRUN_ARGS}SH-IDC1-10-140-24-129"

# python arguments
MODE=$1
CFG=$2
DATA_PATH=$3
PRETRAINED=$4
TAG=$5
BATCH_SIZE=$6
TRAIN_FRAC=$7
MASTER_PORT=${8:-12365}

LAUNCHER="slurm"

PY_ARGS="--cfg ${CFG} \
    --data-path ${DATA_PATH} \
    --pretrained ${PRETRAINED} \
    --tag ${TAG} \
    --launcher ${LAUNCHER} \
    --master_port ${MASTER_PORT}"

if [ "${MODE}" = "pretrain" ]; then
    SCRIPT="main_teacher.py"
    PY_ARGS="${PY_ARGS} --batch-size ${BATCH_SIZE}"
elif [ "${MODE}" = "finetune" ]; then
    SCRIPT="main_finetune.py"
    PY_ARGS="${PY_ARGS} --batch-size ${BATCH_SIZE} --train_frac ${TRAIN_FRAC}"
elif [ "${MODE}" = "test" ]; then
    SCRIPT="main_finetune.py"
    PY_ARGS="${PY_ARGS} --batch-size 128 --test"
    GPUS=1
    GPUS_PER_NODE=$((GPUS/NODE))
fi


TIME=$(date "+%Y%m%d-%H%M%S")
JOB_NAME="GFM_${MODE}[${TIME}]"

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --nodes=$NODE \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python ${SCRIPT} ${PY_ARGS}

# ==================== The command to call this shell script ====================
# finetune
# python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py --cfg configs/BEN.yaml --batch-size 128 \
# --data-path /path/to/bigearthnet/ --pretrained output/simmim_pretrain/gfm.pth --tag BEN --train_frac 0.01

# pretrain
# python -m torch.distributed.launch --nproc_per_node 8 main_teacher.py \
# --cfg configs/simmim_pretrain__swin_base__img192_window6__100ep.yaml --batch-size 128 \
# --data-path /path/to/GeoPileV0/ --pretrained output/simmim_finetune/swin_base_patch4_window7_224_22k.pth --tag gfm
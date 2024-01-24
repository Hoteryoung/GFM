#!/usr/bin/env bash

set -x

# PARTITION="s2_bigdata"
PARTITION="test_s2"

# debug config
# NODE=2
# GPUS=2
# GPUS_PER_NODE=$((GPUS/NODE))
# CPUS_PER_TASK=8

# train config
NODE=2
GPUS=16
GPUS_PER_NODE=$((GPUS/NODE))
CPUS_PER_TASK=8

# using specific nodes
SRUN_ARGS=""
# SRUN_ARGS="--debug"
# SRUN_ARGS="${SRUN_ARGS} --nodelist=SH-IDC1-10-140-24-128,"
# SRUN_ARGS="${SRUN_ARGS}SH-IDC1-10-140-24-129"

# python arguments
BATCH_SIZE=32
FINETUNE_PRETRAIN=$1
CFG=$2
DATA_PATH=$3
PRETRAINED=$4
TAG=$5

LAUNCHER="slurm"

PY_ARGS="--cfg ${CFG} --batch-size ${BATCH_SIZE} --data-path ${DATA_PATH} \
--pretrained ${PRETRAINED} --tag ${TAG} --launcher ${LAUNCHER}"

if [ "${FINETUNE_PRETRAIN}" = "finetune" ]; then
    SCRIPT="main_finetune.py"
    PY_ARGS="${PY_ARGS} --train_frac 0.01"
else
    SCRIPT="main_teacher.py"
fi

TIME=$(date "+%Y%m%d-%H%M%S")

JOB_NAME="GFM_${FINETUNE_PRETRAIN}[${TIME}]"
# JOB_NAME="GFM_pretrain[${TIME}]"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
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


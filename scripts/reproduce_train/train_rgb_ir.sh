#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
echo SCRIPTPATH:${SCRIPTPATH}
PROJECT_DIR="${SCRIPTPATH}/../../"
echo PROJECT_DIR:${PROJECT_DIR}

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=416
# to reproduced the results in our paper, please use:
# TRAIN_IMG_SIZE=840
data_cfg_path="configs/data/rgb_ir_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"

n_nodes=1
n_gpus_per_node=2
torch_num_workers=2
batch_size=6
pin_memory=true
exp_name="outdoor-ds-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))_RGB_IR_homo"

python -u ./train.py \
    --data_cfg_path ${data_cfg_path} \
    --main_cfg_path ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1 \
    --flush_logs_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=30
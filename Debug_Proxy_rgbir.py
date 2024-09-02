# debugProxy.py
import os, sys, runpy

## 1. cd WORKDIR
# os.chdir('WORKDIR')

## 2A. python test.py 4 5


TRAIN_IMG_SIZE=640
TRAIN_IMG_SIZE=416
# to reproduced the results in our paper, please use:
# TRAIN_IMG_SIZE=840
data_cfg_path=f"configs/data/rgb_ir_trainval_{TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"

n_nodes=1
n_gpus_per_node=1
torch_num_workers=0
batch_size=2
pin_memory='true'
exp_name=f"outdoor-ds-{TRAIN_IMG_SIZE}-bs={n_gpus_per_node * n_nodes * batch_size}_debug"
accelerator = "ddp"
args = f"python ./train.py\
    --data_cfg_path {data_cfg_path}\
    --main_cfg_path {main_cfg_path}\
    --exp_name {exp_name}\
    --gpus {n_gpus_per_node} --num_nodes {n_nodes} --accelerator {accelerator}\
    --batch_size {batch_size} --num_workers {torch_num_workers} --pin_memory {pin_memory}\
    --check_val_every_n_epoch 1\
    --log_every_n_steps 1\
    --flush_logs_every_n_steps 1\
    --limit_val_batches 1.\
    --num_sanity_val_steps 10\
    --benchmark True\
    --max_epochs 30"
## 2B. python -m mymodule.test 4 5
# args = 'python -m mymodule.test 4 5'

args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)
if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path
sys.argv.extend(args[1:])
fun(args[0], run_name='__main__')

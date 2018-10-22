#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH
export | grep PYTHONPATH

python tools/train_val.py --config=experiments/config.json \
    --dataset=kitti \
    --datadir=datasets/KITTI/object/ \
    --save_dir=experiments/save \
    --epochs=10 \
    --step_epochs=5,7 \
    --lr=0.1
#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=.
export PYTHONPATH=$ROOT:$PYTHONPATH
export | grep PYTHONPATH

GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/train_val.py --config=experiments/config.json \
    --dataset=kitti \
    --datadir=datasets/KITTI/object/ \
    --save_dir=experiments/save \
    --epochs=10 \
    --step_epochs=7 \
    --lr=0.01 \
    --batch_size=3 \
    --workers=2

#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH

python tools/train_val.py --config=experiments/config.json \
    --dataset=kitti \
    --datadir=datasets/KITTI/object/ \
    --save_dir=experiments/save \
    --resume=experiments/save/checkpoint_e10.pth \
    -e
#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=.
export PYTHONPATH=$ROOT:$PYTHONPATH
export | grep PYTHONPATH

GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/test.py --config=experiments/config.json \
    --dataset=kitti \
    --datadir=datasets/KITTI/ \
    --save_dir=experiments/save \
    --resume=experiments/important/checkpoint_e160_2d_6236.pth \
    -e \
    -v \
    -s \
    --figdir=experiments/save/video_pictures
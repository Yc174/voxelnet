#!/bin/bash

ROOT=..
export PYTHONPATH=$ROOT:$PYTHONPATH
ROOT=.
export PYTHONPATH=$ROOT:$PYTHONPATH
export | grep PYTHONPATH

python tools/visualization_kitti.py --demo

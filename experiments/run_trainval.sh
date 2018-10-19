ROOT=..
python $ROOT/tools/train_val.py --config=/home/yc/Myprojects/voxelnet_yc/voxelnet/experiments/config.json \
    --dataset=kitti \
    --datadir=/home/yc/Myprojects/voxelnet_yc/voxelnet/datasets/KITTI/object/ \
    --save_dir=/home/yc/Myprojects/voxelnet_yc/voxelnet/experiments/save \
    --epochs=10 \
    --step_epochs=5,7 \
    --lr=0.1
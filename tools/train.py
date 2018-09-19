from __future__ import print_function

import os
import json
import argparse

import torch

from lib.dataset.kitti_dataset import KittiDataset, KittiDataloader

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of voxelnet in json format')
parser.add_argument('--dataset', dest='dataset', required=True, choices = ['kitti', 'other'],
                    help='which dataset is used for training')
parser.add_argument('--datadir', dest='datadir', required=True,
                    help='data directory of KITTI when dataset is `kitti`')
args = parser.parse_args()
def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg

def build_data_loader(dataset, cfg):
    if dataset == 'kitti':
        Dataset = KittiDataset
        Dataloader = KittiDataloader

    else:
        pass
    scales = cfg['shared']['scales']
    max_size = cfg['shared']['max_size']
    train_dataset = Dataset(args.datadir)
    train_loader = Dataloader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=False)
    return train_loader

def main():
    cfg = load_config(args.config)
    train_loader = build_data_loader(args.dataset, cfg)
    for iter, input in enumerate(train_loader):
        print('iter%d'%iter, input[1].shape)


if __name__ == "__main__":
    main()
    print("OK!")
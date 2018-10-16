from __future__ import print_function

import os
import json
import argparse

import torch

from lib.dataset.kitti_dataset import KittiDataset, KittiDataloader
from lib.models.voxelnet import Voxelnet

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of voxelnet in json format')
parser.add_argument('--dataset', dest='dataset', required=True, choices = ['kitti', 'other'],
                    help='which dataset is used for training')
parser.add_argument('--datadir', dest='datadir', required=True,
                    help='data directory of KITTI when dataset is `kitti`')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--step_epochs', dest='step_epochs', type=lambda x: list(map(int, x.split(','))),
                    default='-1', help='epochs to decay lr')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

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
    train_dataset = Dataset(args.datadir, cfg)
    train_loader = Dataloader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=False)
    return train_loader

def main():
    cfg = load_config(args.config)
    train_loader = build_data_loader(args.dataset, cfg)
    model = Voxelnet(cfg=cfg)
    print(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model.cuda()

    if args.evaluate:
        validate(train_loader, model, cfg)

    for epoch in range(args.start_epoch, args.epochs):
        if epoch+1 in args.step_epochs:
            lr = adjust_learning_rate(optimizer, 0.1, gradual= True)
        train(train_loader, model, optimizer, epoch, cfg)



def train(dataloader, model, optimizer, epoch, cfg, warmup=False):

    for iter, input in enumerate(dataloader):
        print('iter%d'%iter, input[1].shape)
        x = {
            'cfg': cfg,
            'image': torch.autograd.Variable(input[0]).cuda(),
            'points': input[1],
            'indices': input[2],
            'num_pts': input[3],
            'leaf_out': input[4],
            'voxel_indices': input[5],
            'voxel': torch.autograd.Variable(input[6]).cuda(),
            'gt_bboxes_2d': input[7],
            'gt_bboxes_3d': input[8],
        }
        outputs = model(x)
        rpn_cls_loss = outputs['losses'][0]
        rpn_loc_loss = outputs['losses'][1]
        print('rpn_cls_loss :', rpn_cls_loss)
        print('rpn_loc_loss :', rpn_loc_loss)

        rpn_cls_loss_value = rpn_cls_loss.data[0]
        rpn_loc_loss_value = rpn_loc_loss.data[0]

        loss = rpn_cls_loss + rpn_loc_loss
        loss_value = loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: [%d][%d/%d]'%(epoch, iter, len(dataloader)))

def validate(dataloader, model, cfg):
    # switch to evaluate mode
    model.eval()
    pass

def adjust_learning_rate(optimizer, rate, gradual = True):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = None
    for param_group in optimizer.param_groups:
        if gradual:
            param_group['lr'] *= rate
        else:
            param_group['lr'] = args.lr * rate
        lr = param_group['lr']
    return lr

if __name__ == "__main__":
    main()
    print("OK!")
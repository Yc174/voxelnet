from __future__ import print_function

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import logging
import time
from multiprocessing import Process

from lib.dataset.kitti_dataset_raw_data import KittiDataset, KittiDataloader
from lib.dataset.kitti_util import Calibration
from lib.dataset.kitti_object import show_lidar_with_numpy_boxes
from lib.models.voxelnet import Voxelnet
from lib.functions import log_helper
from lib.functions import bbox_helper
from lib.functions import anchor_projector
from lib.functions import box_3d_encoder
from lib.functions import load_helper
from lib.evaluator import evaluator_utils
import _sys_init


parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of voxelnet in json format')
parser.add_argument('--dataset', dest='dataset', required=True, choices = ['kitti', 'other'],
                    help='which dataset is used for training')
parser.add_argument('--datadir', dest='datadir', required=True,
                    help='data directory of KITTI when dataset is `kitti`')
parser.add_argument('--save_dir', dest='save_dir', default='checkpoints',
                    help='directory to save models')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-v', '--visual', dest='visual', action='store_true',
                    help='visualization detections on point-cloud')
parser.add_argument('-s', '--save_as_figure', dest='save_as_figure', action='store_true',
                    help='whether to save the visualization detections on point-cloud')
parser.add_argument('--figdir', dest='figdir', default='save_figure',
                    help='directory to save results')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--dist', dest='dist', default=1, type=int,
                    help='distributed training or not')
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
    logger = logging.getLogger('global')
    if dataset == 'kitti':
        Dataset = KittiDataset
        Dataloader = KittiDataloader

    else:
        pass
    scales = cfg['shared']['scales']
    max_size = cfg['shared']['max_size']
    val_dataset = Dataset(args.datadir, cfg, split='raw')
    val_loader = Dataloader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    logger.info('build dataloader done')
    return  val_dataset,\
            val_loader

def main():
    log_helper.init_log('global', args.save_dir, logging.INFO)
    logger = logging.getLogger('global')
    cfg = load_config(args.config)
    device = torch.device("cuda:0")
    val_dataset, val_loader = build_data_loader(args.dataset, cfg)
    model = Voxelnet(cfg=cfg)
    logger.info(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, _, best_recall = load_helper.restore_from(model, optimizer, args.resume)

    #model.cuda()

    if torch.cuda.device_count()>1 and args.dist:
       print("Let's use", torch.cuda.device_count(), "GPUs!")
       model = nn.parallel.DistributedDataParallel(model)
    model.to(device)


    if args.evaluate:
        validate(val_dataset, val_loader, model, cfg)
        return


def validate(dataset, dataloader, model, cfg, epoch=-1):
    # switch to evaluate mode
    logger = logging.getLogger('global')
    # torch.cuda.set_device(0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model.eval()

    total_rc = 0
    total_gt = 0
    area_extents = np.asarray(cfg['shared']['area_extents']).reshape(-1, 2)
    bev_extents = area_extents[[0, 2]]

    score_threshold = cfg['test_rpn_proposal_cfg']['score_threshold']
    valid_samples = 0

    logger.info('start validate')
    for iter, _input in enumerate(dataloader):
        gt_boxes = _input[9]
        voxel_with_points = _input[6]
        batch_size = voxel_with_points.shape[0]
        # assert batch_size == 1
        img_ids = _input[10]

        x = {
            'cfg': cfg,
            # 'image': torch.autograd.Variable(_input[0]).cuda(),
            'points': _input[1],
            'indices': _input[2],
            'num_pts': _input[3],
            'leaf_out': _input[4],
            'voxel_indices': _input[5],
            'voxel_points': torch.autograd.Variable(_input[6]).cuda(),
            'ground_plane': _input[7],
            'gt_bboxes_2d': _input[8],
            'gt_bboxes_3d': _input[9],
            'num_divisions': _input[11]
        }

        t0=time.time()
        outputs = model(x)
        outputs = outputs['predict']
        t2 =time.time()
        proposals = outputs[0].data.cpu().numpy()

        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()

        for b_ix in range(batch_size):
            rois_per_points_cloud = proposals[proposals[:, 0] == b_ix]


            if args.visual:

                calib = dataset.kitti.calib

                # Show all LiDAR points. Draw 3d box in LiDAR point cloud
                # print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')

                score_filter = rois_per_points_cloud[:, -1]>score_threshold
                print('img: {}, proposals shape:{}'.format(img_ids[b_ix], rois_per_points_cloud[score_filter].shape))

                show_lidar_with_numpy_boxes(x['points'][b_ix, :, 0:3].numpy(), rois_per_points_cloud[score_filter, 1:1+7][:10],
                                            calib, save_figure=args.save_as_figure, save_figure_dir=args.figdir,
                                            img_name='%s.jpg'%(img_ids[b_ix]),
                                            color=(1, 1, 1))
                # input()
                # anchors = outputs[1]
                # total_anchors, _ = anchors.shape
                # idx = np.random.choice(total_anchors, 200)
                # show_lidar_with_numpy_boxes(x['points'][b_ix, :, 0:3].numpy(), anchors[idx, :], calib,
                #                             color=(1, 1, 1))
                # input()

        log_helper.print_speed(iter + 1, t2 - t0, len(dataloader))

    return

if __name__ == "__main__":
    main()

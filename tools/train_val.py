from __future__ import print_function

import os
import json
import argparse

import numpy as np
import torch
import logging
import time

from lib.dataset.kitti_dataset import KittiDataset, KittiDataloader
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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
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
    logger = logging.getLogger('global')
    if dataset == 'kitti':
        Dataset = KittiDataset
        Dataloader = KittiDataloader

    else:
        pass
    scales = cfg['shared']['scales']
    max_size = cfg['shared']['max_size']
    train_dataset = Dataset(args.datadir, cfg, split='train')
    train_loader = Dataloader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    val_dataset = Dataset(args.datadir, cfg, split='val')
    val_loader = Dataloader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
    logger.info('build dataloader done')
    return train_dataset, val_dataset,\
           train_loader, val_loader

def main():
    log_helper.init_log('global', args.save_dir, logging.INFO)
    logger = logging.getLogger('global')
    cfg = load_config(args.config)
    train_dataset, val_dataset, train_loader, val_loader = build_data_loader(args.dataset, cfg)
    model = Voxelnet(cfg=cfg)
    logger.info(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_recall = load_helper.restore_from(model, optimizer, args.resume)

    model.cuda()

    if args.evaluate:
        validate(val_dataset, val_loader, model, cfg)
        return
    recall = 0
    best_recall = 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch+1 in args.step_epochs:
            lr = adjust_learning_rate(optimizer, 0.1, gradual= True)
        train(train_loader, model, optimizer, epoch, cfg)
        # evaluate on validation set
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            recall = validate(val_dataset, val_loader, model, cfg, epoch=epoch)

        # remember best prec@1 and save checkpoint
        is_best = recall > best_recall
        best_recall = max(recall, best_recall)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_recall': best_recall,
            'optimizer': optimizer.state_dict(),
            }, is_best,
            os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch + 1)))
        logger.info('recall %f(%f)' % (recall, best_recall))



def train(dataloader, model, optimizer, epoch, cfg, warmup=False):
    logger = logging.getLogger('global')
    model.cuda()
    model.train()
    t0 = time.time()
    for iter, _input in enumerate(dataloader):
        lr = adjust_learning_rate(optimizer, 1, gradual=True)
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
        t1 = time.time()
        outputs = model(x)
        rpn_cls_loss = outputs['losses'][0]
        rpn_loc_loss = outputs['losses'][1]
        rpn_accuracy = outputs['accuracy'][0][0] / 100.

        loss = rpn_cls_loss + rpn_loc_loss

        t2 = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t3 = time.time()
        logger.info('Epoch: [%d][%d/%d] LR:%f ForwardTime: %.3f Loss: %0.5f (rpn_cls: %.5f rpn_loc: %.5f rpn_acc: %.5f)'%
                    (epoch, iter, len(dataloader), lr, t2-t1, loss.cpu().data.numpy(), rpn_cls_loss.cpu().data.numpy(), rpn_loc_loss.cpu().data.numpy(), rpn_accuracy.cpu().data.numpy()))
        log_helper.print_speed((epoch - 1) * len(dataloader) + iter + 1, t3 - t0, args.epochs * len(dataloader))
        t0 = t3

def validate(dataset, dataloader, model, cfg, epoch=-1):
    # switch to evaluate mode
    logger = logging.getLogger('global')
    model.eval()

    total_rc = 0
    total_gt = 0
    area_extents = np.asarray(cfg['shared']['area_extents']).reshape(-1, 2)
    bev_extents = area_extents[[0, 2]]

    score_threshold = cfg['test_rpn_proposal_cfg']['score_threshold']
    valid_samples = 0
    native_code_copy = _sys_init.root_dir()+'/experiments/predictions/kitti_native_eval/'
    evaluator_utils.copy_kitti_native_code(native_code_copy)
    predictions_3d_dir = evaluator_utils.get_kitti_predictions(score_threshold, epoch)


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
        outputs = model(x)['predict']
        t2 =time.time()
        proposals = outputs[0].data.cpu().numpy()

        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()

        for b_ix in range(batch_size):

            rois_per_points_cloud = proposals[proposals[:, 0] == b_ix]
            gts_per_points_cloud = gt_boxes[b_ix]
            rois_per_points_cloud_anchor = box_3d_encoder.box_3d_to_anchor(rois_per_points_cloud[:, 1:1 + 7])
            gts_per_points_cloud_anchor = box_3d_encoder.box_3d_to_anchor(gts_per_points_cloud)
            rois_per_points_cloud_bev, _ = anchor_projector.project_to_bev(rois_per_points_cloud_anchor, bev_extents)
            gts_per_points_cloud_bev, _ = anchor_projector.project_to_bev(gts_per_points_cloud_anchor, bev_extents)

            # rpn recall
            num_rc, num_gt = bbox_helper.compute_recall(rois_per_points_cloud_bev, gts_per_points_cloud_bev)
            total_gt += num_gt
            total_rc += num_rc

            if args.visual:
                calib_dir = os.path.join(args.datadir, 'training/calib', '%06d.txt'%(img_ids[b_ix]))
                calib = Calibration(calib_dir)

                # Show all LiDAR points. Draw 3d box in LiDAR point cloud
                print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
                show_lidar_with_numpy_boxes(x['points'][b_ix, :, 0:3].numpy(), gts_per_points_cloud, calib, color=(1,1,1))
                input()
                print('proposals shape:', rois_per_points_cloud.shape)

                show_lidar_with_numpy_boxes(x['points'][b_ix, :, 0:3].numpy(), rois_per_points_cloud[:10, 1:1+7], calib,
                                            color=(1, 1, 1))
                input()
                # anchors = outputs[1]
                # total_anchors, _ = anchors.shape
                # idx = np.random.choice(total_anchors, 100)
                # show_lidar_with_numpy_boxes(x['points'][b_ix, :, 0:3].numpy(), anchors[idx, :], calib,
                #                             color=(1, 1, 1))
                # input()

            valid, total_samples= evaluator_utils.save_predictions_in_kitti_format(dataset, rois_per_points_cloud[:,1:], img_ids[b_ix], predictions_3d_dir, score_threshold)
            valid_samples += valid
            logger.info('valid samples: %d/%d'%(valid_samples, total_samples))
        logger.info('Test valid instance: [%d/%d] Time: %.3f %d/%d' % (iter, len(dataloader), t2 - t0, total_rc, total_gt))
        log_helper.print_speed(iter + 1, t2 - t0, len(dataloader))

    logger.info('rpn300 recall=%f'% (total_rc/total_gt))
    evaluate_name = dataset.id2names[1]+'_'+dataset.split
    evaluator_utils.run_kitti_native_script(native_code_copy, evaluate_name, score_threshold, epoch)
    evaluator_utils.run_kitti_native_script_with_05_iou(native_code_copy, evaluate_name, score_threshold, epoch)
    return total_rc/total_gt

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth')

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
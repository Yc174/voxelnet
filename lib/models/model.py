import torch.nn.functional as F
import torch.nn as nn
import torch
import functools

from lib.functions import anchor_target_3d

class model(nn.Module):
    def __init__(self, cfg):
        super(model, self).__init__()

    def feature_extractor(self, voxel_with_points, num_pts, leaf_out, voxel_indices):
        raise NotImplementedError

    def rpn(self, x):
        raise NotImplementedError

    def rcnn(self, x, rois):
        pass

    def _add_rpn_loss(self, compute_anchor_targets_fn, rpn_pred_cls,
                      rpn_pred_loc):
        '''
        :param compute_anchor_targets_fn: functions to produce anchors' learning targets.
        :param rpn_pred_cls: [B, num_anchors * 2, h, w], output of rpn for classification.
        :param rpn_pred_loc: [B, num_anchors * 4, h, w], output of rpn for localization.
        :return: loss of classification and localization, respectively.
        '''
        # [B, num_anchors * 2, h, w], [B, num_anchors * 4, h, w]
        cls_targets, loc_targets, loc_masks, loc_normalizer = \
                compute_anchor_targets_fn(rpn_pred_loc.size())

        # tranpose to the input format of softmax_loss function
        rpn_pred_cls = rpn_pred_cls.permute(0,2,3,1).contiguous().view(-1, 2)
        cls_targets = cls_targets.permute(0,2,3,1).contiguous().view(-1)
        rpn_loss_cls = F.cross_entropy(
            rpn_pred_cls, cls_targets, ignore_index=-1)
        # mask out negative anchors
        rpn_loss_loc = smooth_l1_loss_with_sigma(rpn_pred_loc * loc_masks,
                                                 loc_targets) / loc_normalizer

        # classification accuracy, top1
        acc = accuracy(rpn_pred_cls.data, cls_targets.data)[0]
        return rpn_loss_cls, rpn_loss_loc, acc

    def _pin_args_to_fn(self, cfg, ground_truth_bboxes, image_info, ignore_regions):
        partial_fn = {}
        if self.training:
            partial_fn['anchor_target_fn'] = functools.partial(
                anchor_target_3d.compute_anchor_targets,
                cfg=cfg['train_anchor_target_cfg'],
                ground_truth_bboxes=ground_truth_bboxes,
                ignore_regions=ignore_regions,
                image_info=image_info)
        #     partial_fn['proposal_target_fn'] = functools.partial(
        #         compute_proposal_targets,
        #         cfg=cfg['train_proposal_target_cfg'],
        #         ground_truth_bboxes=ground_truth_bboxes,
        #         ignore_regions=ignore_regions,
        #         image_info=image_info)
        #     partial_fn['rpn_proposal_fn'] = functools.partial(
        #         compute_rpn_proposals,
        #         cfg=cfg['train_rpn_proposal_cfg'],
        #         image_info=image_info)
        # else:
        #     partial_fn['rpn_proposal_fn'] = functools.partial(
        #         compute_rpn_proposals,
        #         cfg=cfg['test_rpn_proposal_cfg'],
        #         image_info=image_info)
        #     partial_fn['predict_bbox_fn'] = functools.partial(
        #         compute_predicted_bboxes,
        #         image_info = image_info,
        #         cfg=cfg['test_predict_bbox_cfg'])
        return partial_fn

    def forward(self, input):
        cfg = input['cfg']
        images = input['image']
        points = input['points']
        indices = input['indices']
        num_pts = input['num_pts']
        leaf_out = input['leaf_out']
        voxel_indices = input['voxel_indices']
        voxel_with_points = input['voxel']
        gt_bboxes_2d = input['gt_bboxes_2d']
        gt_bboxes_3d = input['gt_bboxes_3d']

        partial_fn = self._pin_args_to_fn(
                cfg,
                gt_bboxes_3d,
                image_info=None,
                ignore_regions=None)

        outputs = {'losses': [], 'predict': [], 'accuracy': []}
        x = self.feature_extractor(voxel_with_points, num_pts, leaf_out, voxel_indices)
        rpn_pred_cls, rpn_pred_loc = self.rpn(x)
        print("rpn_pred_cls shape:", rpn_pred_cls.size())
        print("rpn_pred_loc shape:", rpn_pred_loc.size())

        if self.training:
            # train rpn
            rpn_loss_cls, rpn_loss_loc, rpn_acc = \
                    self._add_rpn_loss(partial_fn['anchor_target_fn'],
                            rpn_pred_cls,
                            rpn_pred_loc)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc]
            outputs['accuracy'] = [rpn_acc]
            outputs['predict'] = None
        return outputs

def smooth_l1_loss_with_sigma(pred, targets, sigma=3.0):
    sigma_2 = sigma**2
    diff = pred - targets
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    loss = torch.pow(diff, 2) * sigma_2 / 2. * smoothL1_sign \
            + abs_diff - 0.5 / sigma_2 * (1. - smoothL1_sign)
    reduced_loss = torch.sum(loss)
    return reduced_loss
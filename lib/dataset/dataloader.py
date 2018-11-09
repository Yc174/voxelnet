''' Prepare KITTI data for 3D object detection.

Author: carlyan
Date: October 2018
'''

import os
import sys
import numpy as np
import cv2
import time

from kitti_object import *
import experiments.config as cfg

class Dataloader():
    def __init__(self, cfg, split = 'training', random=False):
        self.data = []
        self.cfg = cfg
        self._random = random

        self.dataset = kitti_object(os.path.join(cfg.DATA_DIR, 'KITTI/object'), split)
        if split == 'training':
            idx_filename = 'train.txt'
        self.data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # If the random flag is set,
        # then the database is shuffled according to system time
        # Useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        self._perm = np.random.permutation(np.arange(len(self.data_idx_list)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the indices for the next minibatch."""

        if self._cur + self.cfg.TRAIN.DATA_PER_BATCH >= len(self.data_idx_list):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self.cfg.TRAIN.DATA_PER_BATCH]
        self._cur += self.cfg.TRAIN.DATA_PER_BATCH

        return db_inds

    def get_minibatch(self, minibatch):
        """Given a minibatch, construct a minibatch sampled from it."""
        per_data = {}
        minibatch_data = []
        for i in minibatch:
            calib = self.dataset.get_calibration(i)  # 3 by 4 matrix
            objects = self.dataset.get_label_objects(i)
            pc_velo = self.dataset.get_lidar(i)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            img = self.dataset.get_image(i)
            img_height, img_width, img_channel = img.shape
            imgfov_pc_velo, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                                  calib, 0, 0, img_width, img_height,
                                                                                  True)

            per_data['calib'] = calib
            per_data['objects'] = objects
            per_data['points'] = imgfov_pc_velo
            per_data['image'] = img

            minibatch_data.append(per_data)

        return minibatch_data

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
  
        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self.data_idx_list[i] for i in db_inds]
        return self.get_minibatch(minibatch_db)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

if __name__ == '__main__':
    cfg = cfg.cfg
    data = Dataloader(cfg, split="training")
    for i in range(10):
        minibatch = data.forward()
        print('*'*20, 'the %d batch'%(i), '*'*20)
        print(minibatch)
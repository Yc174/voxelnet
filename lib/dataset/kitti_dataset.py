import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os

from lib.dataset.kitti_object import kitti_object, get_lidar_in_image_fov
import lib.dataset.kitti_util as utils

class KittiDataset(Dataset):
    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.kitti = kitti_object(root_dir, split)

        if split == 'training':
            idx_filename = 'train.txt'
            idx_filename = os.path.join(os.path.dirname(__file__), idx_filename)
        self.img_ids = [int(line.rstrip()) for line in open(idx_filename)]

        self.names = ['car', 'pedestrian', 'cyclist']
        self.num = len(self.img_ids)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.kitti.get_image(img_id)
        label_objects = self.kitti.get_label_objects(img_id)
        calib = self.kitti.get_calibration(img_id)
        pc_velo = self.kitti.get_lidar(img_id)
        bboxes_2d = []
        bboxes_3d = []
        for object in label_objects:
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(object, calib.P)
            bboxes_2d.append(object.box2d)
            bboxes_3d.append(box3d_pts_3d)
        bboxes_2d = np.array(bboxes_2d)
        bboxes_3d = np.array(bboxes_3d)

        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img_height, img_width, img_channel = img.shape
        imgfov_pc_velo, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                              calib, 0, 0, img_width, img_height,
                                                                              True)
        imgfov_pc_rect = pc_rect[img_fov_inds]
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        return [img.unsqueeze(0),
                bboxes_2d,
                bboxes_3d,
                [img_height, img_width],
                imgfov_pc_rect]

class KittiDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False):
        super(KittiDataloader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                            num_workers, self._collate_fn, pin_memory, drop_last)

    def _collate_fn(self, batch):
        batch_size = len(batch)
        zip_batch = list(zip(*batch))
        images = zip_batch[0]
        ground_truth_bboxes_2d = zip_batch[1]
        ground_truth_bboxes_3d = zip_batch[2]
        img_info = zip_batch[3]
        imgfov_pc_velo = zip_batch[4]

        max_img_h = max([_.shape[-2] for _ in images])
        max_img_w = max([_.shape[-1] for _ in images])
        max_num_gt_bboxes_2d = max([_.shape[0] for _ in ground_truth_bboxes_2d])
        max_num_gt_bboxes_3d = max([_.shape[0] for _ in ground_truth_bboxes_3d])
        max_points = max([_.shape[0] for _ in imgfov_pc_velo])

        padded_images = []
        padded_gt_bboxes_2d = []
        padded_gt_bboxes_3d = []
        padded_points = []
        for b_ix in range(batch_size):
            img = images[b_ix]
            # pad zeros to right bottom of each image
            pad_size = (0, max_img_w - img.shape[-1], 0, max_img_h - img.shape[-2])
            padded_images.append(F.pad(img, pad_size, 'constant', 0).data.cpu())

            # pad zeros to gt_bboxes
            gt_bboxes_2d = ground_truth_bboxes_2d[b_ix]
            new_gt_bboxes_2d = np.zeros([max_num_gt_bboxes_2d, gt_bboxes_2d.shape[-1]])
            new_gt_bboxes_2d[range(gt_bboxes_2d.shape[0]), :] = gt_bboxes_2d
            padded_gt_bboxes_2d.append(new_gt_bboxes_2d)

            gt_bboxes_3d = ground_truth_bboxes_3d[b_ix]
            new_gt_bboxes_3d = np.zeros([max_num_gt_bboxes_3d, gt_bboxes_3d.shape[-2], gt_bboxes_3d.shape[-1]])
            new_gt_bboxes_3d[range(gt_bboxes_3d.shape[0]), :, :] = gt_bboxes_3d
            padded_gt_bboxes_3d.append(new_gt_bboxes_3d)

            points = imgfov_pc_velo[b_ix]
            new_points = np.zeros([max_points, points.shape[-1]])
            new_points[range(points.shape[0]), :] = points
            padded_points.append(new_points)

        padded_images = torch.cat(padded_images, dim = 0)
        padded_gt_bboxes_2d = torch.from_numpy(np.stack(padded_gt_bboxes_2d, axis = 0))
        padded_gt_bboxes_3d = torch.from_numpy(np.stack(padded_gt_bboxes_3d, axis = 0))
        padded_points = torch.from_numpy(np.stack(padded_points, axis= 0))
        return padded_images, padded_points, padded_gt_bboxes_2d, padded_gt_bboxes_3d

def test(root_dir):
    kitti = KittiDataset(root_dir=root_dir, split='training')
    loader = KittiDataloader(kitti, batch_size=2, shuffle=False, num_workers=2)
    for iter, input in enumerate(loader):
        imgs = input[0]
        padded_gt_bboxes = input[2]
        print('iter%d, img shape:'%iter, imgs.shape)

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '../..', 'datasets/KITTI/object')

    test(root_dir=data_dir)
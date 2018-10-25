import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time

from lib.dataset.kitti_object import kitti_object, get_lidar_in_image_fov, get_lidar_in_area_extent, get_lidar_in_img_fov_and_area_extent
import lib.dataset.kitti_util as utils
from lib.dataset.voxel_grid import VoxelGrid

class KittiDataset(Dataset):
    def __init__(self, root_dir, cfg, split='train'):
        self.root_dir = root_dir
        self.config = cfg
        self.voxel_size = cfg['shared']['voxel_size']
        area_extents = cfg['shared']['area_extents']
        self.area_extents = np.array(area_extents).reshape(3, 2)
        idx_filename = ''
        if split == 'train':
            # idx_filename = 'train.txt'
            idx_filename = 'generated_Car_training.txt'
            split = 'training'
        elif split == 'val':
            idx_filename = 'val.txt'
            split = 'training' # rename
        elif split == 'test':
            idx_filename = 'test.txt'
            split = 'testing'
        idx_filename = os.path.join(os.path.dirname(__file__), 'idx_files', idx_filename)
        self.img_ids = [int(line.rstrip()) for line in open(idx_filename)]

        self.kitti = kitti_object(root_dir, split)
        # self.names = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
        self.names = {'Car': 1}
        self.num = len(self.img_ids)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        t0 = time.time()
        img_id = self.img_ids[idx]
        img = self.kitti.get_image(img_id)
        label_objects = self.kitti.get_label_objects(img_id)
        calib = self.kitti.get_calibration(img_id)
        pc_velo = self.kitti.get_lidar(img_id)
        ground_plane = self.kitti.get_ground_plane(img_id)
        t1 = time.time()
        bboxes_2d = []
        bboxes_3d = []
        for object in label_objects:
            box3d = utils.object_label_to_box_3d(object)
            if object.type not in self.names.keys():
                continue
            # ty = self.names[object.type]
            # box3d = np.append(box3d, ty)
            bboxes_2d.append(object.box2d)
            bboxes_3d.append(box3d)
        bboxes_2d = np.asarray(bboxes_2d).reshape(-1, 4)
        bboxes_3d = np.asarray(bboxes_3d).reshape(-1, 7)
        # print(bboxes_3d[:, 3:])
        t2 = time.time()
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img_height, img_width, img_channel = img.shape
        imgfov_pc_velo, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                              calib, 0, 0, img_width, img_height,
                                                                              True)
        _, area_inds = get_lidar_in_area_extent(pc_velo[:, :3], calib, self.area_extents)
        inds = (area_inds & img_fov_inds)
        # inds = get_lidar_in_img_fov_and_area_extent(pc_velo[:, :3], pc_rect[:, :3],
        #                                             calib, 0, 0, img_width, img_height,self.area_extents )
        # print("points number in area_extents: %d"%(len(area_inds[area_inds==1])))
        # print("points number in img_fov: %d"%len(img_fov_inds[img_fov_inds==1]))
        # print("points left: %d"%len(inds[inds==1]))
        imgfov_pc_rect = pc_rect[inds]
        t3 = time.time()
        voxel_grid = VoxelGrid()
        voxel_grid.voxelize(imgfov_pc_rect, voxel_size=self.voxel_size, extents=self.area_extents, create_leaf_layout=True)
        t4 = time.time()
        # to_tensor = transforms.ToTensor()
        # img = to_tensor(img)
        print('sys used time, load data: %.5f, get gt: %.5f,get valid points: %.5f, voxel_grid: %.5f'%(t1-t0, t2-t1, t3-t2,t4-t3))
        return [None,
                bboxes_2d,
                bboxes_3d,
                [img_height, img_width],
                voxel_grid.points,
                voxel_grid.unique_indices,
                voxel_grid.num_pts_in_voxel,
                voxel_grid.leaf_layout,
                voxel_grid.voxel_indices,
                voxel_grid.padded_voxel_points,
                ground_plane,
                img_id,
                voxel_grid.num_divisions
                ]
        # return [img.unsqueeze(0),
        #         bboxes_2d,
        #         bboxes_3d,
        #         [img_height, img_width],
        #         voxel_grid.points,
        #         voxel_grid.unique_indices,
        #         voxel_grid.num_pts_in_voxel,
        #         voxel_grid.leaf_layout,
        #         voxel_grid.voxel_indices,
        #         voxel_grid.padded_voxel_points,
        #         ground_plane,
        #         img_id,
        #         voxel_grid.num_divisions
        #         ]

class KittiDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False):
        super(KittiDataloader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                            num_workers, self._collate_fn, pin_memory, drop_last)

    def _collate_fn(self, batch):
        batch_size = len(batch)
        zip_batch = list(zip(*batch))
        # images = zip_batch[0]
        ground_truth_bboxes_2d = zip_batch[1]
        ground_truth_bboxes_3d = zip_batch[2]
        img_info = zip_batch[3]
        s_points = zip_batch[4]
        unique_indices = zip_batch[5]
        num_pts_in_voxel = zip_batch[6]
        leaf_out = zip_batch[7]
        s_voxel_indices = zip_batch[8]
        s_voxel_points = zip_batch[9]
        ground_plane = zip_batch[10]
        img_ids = zip_batch[11]
        num_divisions = zip_batch[12]

        # max_img_h = max([_.shape[-2] for _ in images])
        # max_img_w = max([_.shape[-1] for _ in images])
        max_num_gt_bboxes_2d = max([_.shape[0] for _ in ground_truth_bboxes_2d])
        max_num_gt_bboxes_3d = max([_.shape[0] for _ in ground_truth_bboxes_3d])
        max_points = max([_.shape[0] for _ in s_points])
        max_indices = max([_.shape[0] for _ in unique_indices])
        # max_num_pts = max([_.shape[0] for _ in num_pts_in_voxel])
        # max_voxel_indices = max([_.shape[0] for _ in s_voxel_indices])

        padded_images = []
        padded_gt_bboxes_2d = []
        padded_gt_bboxes_3d = []
        padded_points = []
        padded_indices = []
        padded_num_pts = []
        padded_voxel_indices = []
        padded_voxel_points = []

        for b_ix in range(batch_size):
            # img = images[b_ix]
            # # pad zeros to right bottom of each image
            # pad_size = (0, max_img_w - img.shape[-1], 0, max_img_h - img.shape[-2])
            # padded_images.append(F.pad(img, pad_size, 'constant', 0).data.cpu())

            # pad zeros to gt_bboxes
            gt_bboxes_2d = ground_truth_bboxes_2d[b_ix]
            new_gt_bboxes_2d = np.zeros([max_num_gt_bboxes_2d, gt_bboxes_2d.shape[-1]])
            new_gt_bboxes_2d[:gt_bboxes_2d.shape[0], :] = gt_bboxes_2d
            padded_gt_bboxes_2d.append(new_gt_bboxes_2d)

            gt_bboxes_3d = ground_truth_bboxes_3d[b_ix]
            new_gt_bboxes_3d = np.zeros([max_num_gt_bboxes_3d, gt_bboxes_3d.shape[-1]])
            new_gt_bboxes_3d[:gt_bboxes_3d.shape[0], :] = gt_bboxes_3d
            padded_gt_bboxes_3d.append(new_gt_bboxes_3d)

            points = s_points[b_ix]
            new_points = np.zeros([max_points, points.shape[-1]])
            new_points[:points.shape[0], :] = points
            padded_points.append(new_points)

            indices = unique_indices[b_ix]
            new_indices = np.zeros([max_indices])
            new_indices[:indices.shape[0]] = indices
            padded_indices.append(new_indices)

            num_pts = num_pts_in_voxel[b_ix]
            new_num_pts = np.zeros(max_indices)
            new_num_pts[:num_pts.shape[0]] = num_pts
            padded_num_pts.append(new_num_pts)

            voxel_indices = s_voxel_indices[b_ix]
            new_voxel_indices = np.zeros([max_indices, voxel_indices.shape[-1]+1], dtype=np.int64)
            new_voxel_indices[:voxel_indices.shape[0], 1:] = voxel_indices
            new_voxel_indices[:voxel_indices.shape[0], 0] = b_ix
            padded_voxel_indices.append(new_voxel_indices)

            voxel_points = s_voxel_points[b_ix]
            new_voxel_points = np.zeros([max_indices, voxel_points.shape[-2], voxel_points.shape[-1]], dtype=np.float32)
            new_voxel_points[:voxel_points.shape[0], :] = voxel_points
            padded_voxel_points.append(new_voxel_points)

        # padded_images = torch.cat(padded_images, dim = 0)
        padded_gt_bboxes_2d = torch.from_numpy(np.stack(padded_gt_bboxes_2d, axis = 0))
        padded_gt_bboxes_3d = torch.from_numpy(np.stack(padded_gt_bboxes_3d, axis = 0))
        padded_points = torch.from_numpy(np.stack(padded_points, axis= 0))
        padded_indices = torch.from_numpy(np.stack(padded_indices, axis= 0))
        padded_num_pts = torch.from_numpy(np.stack(padded_num_pts, axis= 0))
        leaf_out = torch.from_numpy(np.array(leaf_out))
        padded_voxel_indices = torch.from_numpy(np.stack(padded_voxel_indices, axis=0))
        # voxel = torch.from_numpy(np.array(s_voxel))
        ground_plane = torch.from_numpy(np.array(ground_plane))
        padded_voxel_points = torch.from_numpy(np.stack(padded_voxel_points, axis=0))
        num_divisions = np.asarray(num_divisions)

        # print("padded img size:", padded_images.size())
        # print("padded gt_bboxes_3d size", padded_gt_bboxes_3d.size())
        # print("padded points size:", padded_points.size())
        # print("padded indices size:", padded_indices.size())
        # print("padded num_pts size:", padded_num_pts.size())
        # print("leaf_out size:", leaf_out.size())
        # print("padded voxel indices:", padded_voxel_indices.size())
        # print("voxel shape:", voxel.size())
        # print('ground_plane shape:', ground_plane.size())
        # print('img_ids :', img_ids)
        # print("padded voxel_points shape:", padded_voxel_points.size())
        # print("num_divisions:", num_divisions)

        return padded_images, padded_points, padded_indices, padded_num_pts, \
               leaf_out, padded_voxel_indices, padded_voxel_points, ground_plane, \
               padded_gt_bboxes_2d, padded_gt_bboxes_3d, img_ids, num_divisions


def load_config(config_path):
    assert(os.path.exists(config_path))
    import json
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg

def test(root_dir):
    cfg_file = os.path.join(os.path.dirname(__file__), '../..', 'experiments/config.json')
    cfg = load_config(cfg_file)
    kitti = KittiDataset(root_dir=root_dir, cfg=cfg, split='train')
    loader = KittiDataloader(kitti, batch_size=2, shuffle=False, num_workers=1)
    t0 = time.time()
    for iter, _input in enumerate(loader):
        points = torch.autograd.Variable(_input[6]).cuda(),
        t2 =time.time()
        print('iter%d, time: %.5f s/iter'%(iter, t2-t0))
        t0 = t2

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '../..', 'datasets/KITTI/object')

    test(root_dir=data_dir)
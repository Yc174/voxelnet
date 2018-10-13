from .model import model
from .torch_util import Conv2d, Conv3d
from .region_proposal_network import RPN
import torch.nn.functional as F
import torch.nn as nn
import torch


class FCN(nn.Module):
    def __init__(self, inplanes, planes):
        super(FCN, self).__init__()
        planes = int(planes/2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,  stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class VFE(nn.Module):
    def __init__(self, inplanes, planes):
        super(VFE, self).__init__()
        self.fcn1 = FCN(inplanes, planes)

    def forward(self, x):
        batch, channel, voxels, num_T = x.size()
        out = self.fcn1(x)
        point_wise_feature = F.max_pool2d(out, kernel_size=[1, num_T], stride=[1, num_T])
        # print('point_wise_feature size:', point_wise_feature.size())
        out = torch.cat((out, point_wise_feature.repeat(1, 1, 1, num_T)), 1)
        # print('VFE size:', out.size())
        return out

class Conv_Middle_layers(nn.Module):
    def __init__(self, ):
        super(Conv_Middle_layers, self).__init__()
        self.conv1 = Conv3d(128, 64, stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv2 = Conv3d(64, 64, stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3 = Conv3d(64, 64, stride=(2, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        shape = out.size()
        print("conv3d feature size:", shape)
        out = out.view(shape[0], -1, shape[-2], shape[-1])
        print("after reshape size:", out.size())
        return out

class feature_learning_network(nn.Module):
    def __init__(self):
        super(feature_learning_network, self).__init__()
        self.vfe1 = VFE(4, 32)
        self.fcn1 = FCN(32, 256)

    def forward(self, x):
        batch, channel, voxels, num_T = x.size()
        out = self.vfe1(x)
        out = self.fcn1(out)
        point_wise_feature = F.max_pool2d(out, kernel_size=[1, num_T], stride=[1, num_T])
        return point_wise_feature

class Voxelnet(model):
    def __init__(self, cfg):
        super(Voxelnet, self).__init__(cfg=cfg)
        self.number_T = cfg['shared']['number_T']
        self.use_random_sampling = cfg['shared']['use_random_sampling']
        self.num_anchors = cfg['shared']['num_anchors']
        self.num_classes = cfg['shared']['num_classes']

        self.feature_learnig = feature_learning_network()
        self.conv3d = Conv_Middle_layers()
        self._rpn = RPN(self.num_classes, self.num_anchors)

    def RandomSampleing(self):
        pass

    def feature_extractor(self, voxel_with_points, num_pts, leaf_out, voxel_indices):
        batch, channel, z, y, x, num_T, = voxel_with_points.size()
        reshaped_voxel_with_points = voxel_with_points.view(batch, channel, y*z*x, num_T)
        print("voxel_with_points size: ", voxel_with_points.size())
        print("reshaped_voxel_with_points size:", reshaped_voxel_with_points.size())

        features = self.feature_learnig(reshaped_voxel_with_points)
        features = features.view(batch, -1, z, y, x)
        print('features learing size:', features.size())
        out = self.conv3d(features)

        return out

    def rpn(self, x):
        rpn_pred_cls, rpn_pred_loc = self._rpn(x)
        return rpn_pred_cls, rpn_pred_loc

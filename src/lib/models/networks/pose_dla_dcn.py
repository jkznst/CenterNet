from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .DCNv2.dcn_v2 import DCN, DCNFA, DCNv2

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.residual = None
        if inplanes != planes:
            self.residual = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None):
        if residual is None:
            residual = self.residual(x) if self.residual else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class CrossStageAggregation(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CrossStageAggregation, self).__init__()
        self.base_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.base_bn = nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM)

        self.dla_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.dla_bn = nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, base_feat, dla_feat, second_stage_feat=None):
        out = self.base_conv(base_feat)
        out = self.base_bn(out)
        out = self.relu(out)

        out_1 = self.relu(self.dla_bn(self.dla_conv(dla_feat)))
        out = out + out_1
        if second_stage_feat is not None:
            out += second_stage_feat
        return out

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.stride8_to_4_conv = nn.Conv2d(in_channels[1], out_channels[0], kernel_size=1, bias=False)
        self.stride8_to_4_bn = nn.BatchNorm2d(out_channels[0], momentum=BN_MOMENTUM)
        self.stride8_to_4_up = nn.Upsample(scale_factor=2, mode='nearest')

        self.stride16_to_4_conv = nn.Conv2d(in_channels[2], out_channels[0], kernel_size=1, bias=False)
        self.stride16_to_4_bn = nn.BatchNorm2d(out_channels[0], momentum=BN_MOMENTUM)
        self.stride16_to_4_up = nn.Upsample(scale_factor=4, mode='nearest')

        self.stride32_to_4_conv = nn.Conv2d(in_channels[3], out_channels[0], kernel_size=1, bias=False)
        self.stride32_to_4_bn = nn.BatchNorm2d(out_channels[0], momentum=BN_MOMENTUM)
        self.stride32_to_4_up = nn.Upsample(scale_factor=8, mode='nearest')

        ########
        self.stride4_to_8_conv = nn.Conv2d(in_channels[0], out_channels[1], kernel_size=3, stride=2,
                                           padding=1, bias=False)
        self.stride4_to_8_bn = nn.BatchNorm2d(out_channels[1], momentum=BN_MOMENTUM)

        self.stride16_to_8_conv = nn.Conv2d(in_channels[2], out_channels[1], kernel_size=1, bias=False)
        self.stride16_to_8_bn = nn.BatchNorm2d(out_channels[1], momentum=BN_MOMENTUM)
        self.stride16_to_8_up = nn.Upsample(scale_factor=2, mode='nearest')

        self.stride32_to_8_conv = nn.Conv2d(in_channels[3], out_channels[1], kernel_size=1, bias=False)
        self.stride32_to_8_bn = nn.BatchNorm2d(out_channels[1], momentum=BN_MOMENTUM)
        self.stride32_to_8_up = nn.Upsample(scale_factor=4, mode='nearest')

        #########
        self.stride4_to_16_conv0 = nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=2,
                                           padding=1, bias=False)
        self.stride4_to_16_bn0 = nn.BatchNorm2d(in_channels[0], momentum=BN_MOMENTUM)
        self.stride4_to_16_conv1 = nn.Conv2d(in_channels[0], out_channels[2], kernel_size=3, stride=2,
                                           padding=1, bias=False)
        self.stride4_to_16_bn1 = nn.BatchNorm2d(out_channels[2], momentum=BN_MOMENTUM)

        self.stride8_to_16_conv = nn.Conv2d(in_channels[1], out_channels[2], kernel_size=3, stride=2,
                                           padding=1, bias=False)
        self.stride8_to_16_bn = nn.BatchNorm2d(out_channels[2], momentum=BN_MOMENTUM)

        self.stride32_to_16_conv = nn.Conv2d(in_channels[3], out_channels[2], kernel_size=1, bias=False)
        self.stride32_to_16_bn = nn.BatchNorm2d(out_channels[2], momentum=BN_MOMENTUM)
        self.stride32_to_16_up = nn.Upsample(scale_factor=2, mode='nearest')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_feat_list):
        inp_stride4, inp_stride8, inp_stride16, inp_stride32 = input_feat_list
        out_stride4 = inp_stride4
        out_stride4_path0 = self.stride8_to_4_up(self.stride8_to_4_bn(self.stride8_to_4_conv(inp_stride8)))
        out_stride4_path1 = self.stride16_to_4_up(self.stride16_to_4_bn(self.stride16_to_4_conv(inp_stride16)))
        out_stride4_path2 = self.stride32_to_4_up(self.stride32_to_4_bn(self.stride32_to_4_conv(inp_stride32)))
        out_stride4 = out_stride4 + out_stride4_path0 + out_stride4_path1 + out_stride4_path2
        out_stride4 = self.relu(out_stride4)

        out_stride8 = inp_stride8
        out_stride8_path0 = self.stride4_to_8_bn(self.stride4_to_8_conv(inp_stride4))
        out_stride8_path1 = self.stride16_to_8_up(self.stride16_to_8_bn(self.stride16_to_8_conv(inp_stride16)))
        out_stride8_path2 = self.stride32_to_8_up(self.stride32_to_8_bn(self.stride32_to_8_conv(inp_stride32)))
        out_stride8 = out_stride8 + out_stride8_path0 + out_stride8_path1 + out_stride8_path2
        out_stride8 = self.relu(out_stride8)

        out_stride16 = inp_stride16
        out_stride16_path0 = self.stride4_to_16_bn1(self.stride4_to_16_conv1(self.relu(self.stride4_to_16_bn0(self.stride4_to_16_conv0(inp_stride4)))))
        out_stride16_path1 = self.stride8_to_16_bn(self.stride8_to_16_conv(inp_stride8))
        out_stride16_path2 = self.stride32_to_16_up(self.stride32_to_16_bn(self.stride32_to_16_conv(inp_stride32)))
        out_stride16 = out_stride16 + out_stride16_path0 + out_stride16_path1 + out_stride16_path2
        out_stride16 = self.relu(out_stride16)
        return [out_stride4, out_stride8, out_stride16]

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            # print(i)
            # print(x.size())
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

def dla60(pretrained=True, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla60', hash='24839fc4')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class FeatureAdaptation(nn.Module):
    def __init__(self, chi, cho):
        super(FeatureAdaptation, self).__init__()
        self.conv_offset = nn.Conv2d(1,
                                    self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=(0, 0),
                                    bias=True)
        self.conv_mask = nn.Conv2d(1,
                                    self.deformable_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=(0, 0),
                                    bias=True)

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()

        self.conv = DCNv2(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x, centerness, scale):
        mask = self.conv_mask(centerness)
        offset = self.conv_offset(scale)
        mask = torch.sigmoid(mask)
        x = self.conv(x, offset, mask)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            # proj = nn.Sequential(
            #     nn.Conv2d(c, o,
            #               kernel_size=3, stride=1,
            #               padding=1, bias=False),
            #     nn.BatchNorm2d(o, momentum=BN_MOMENTUM),
            #     nn.ReLU(inplace=True))
            node = DeformConv(o, o)
            # node = nn.Sequential(
            #     nn.Conv2d(o, o,
            #               kernel_size=3, stride=1,
            #               padding=1, bias=False),
            #     nn.BatchNorm2d(o, momentum=BN_MOMENTUM),
            #     nn.ReLU(inplace=True))
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out  # [4s, 8s, 16s, 32s]


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
            # z.append(self.__getattr__(head)(y[-1]))
        return [z]

class TwoStageDLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(TwoStageDLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        # second stage
        self.second_stage_bottleneck0 = Bottleneck(inplanes=channels[2], planes=channels[3], stride=2, dilation=1)
        self.second_stage_bottleneck1 = Bottleneck(inplanes=channels[3], planes=channels[4], stride=2, dilation=1)
        self.second_stage_bottleneck2 = Bottleneck(inplanes=channels[4], planes=channels[5], stride=2, dilation=1)

        self.second_stage_csa0 = CrossStageAggregation(in_channel=channels[2], out_channel=channels[2])
        self.second_stage_csa1 = CrossStageAggregation(in_channel=channels[3], out_channel=channels[3])
        self.second_stage_csa2 = CrossStageAggregation(in_channel=channels[4], out_channel=channels[4])
        self.second_stage_csa3 = CrossStageAggregation(in_channel=channels[5], out_channel=channels[5])

        # self.second_stage_feature_fusion = FeatureFusion(in_channels=channels[self.first_level:],
        #                                                  out_channels=channels[self.first_level:self.last_level])
        self.second_stage_dla_up = DLAUp(startp=0, channels=channels[self.first_level:], scales=scales)

        self.second_stage_ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.second_stage_conv0 = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        fill_fc_weights(self.second_stage_conv0)
        self.second_stage_conv1 = nn.Sequential(
            nn.Conv2d(head_conv, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        fill_fc_weights(self.second_stage_conv1)
        self.sigmoid = nn.Sigmoid()
        # self.feature_adaptation = DCNFA(channels[self.first_level], channels[self.first_level],
        #                               kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
        self.feature_adaptation = FeatureAdaptation(channels[self.first_level], channels[self.first_level])

        # self.second_stage_dcn2 = DeformConv(channels[self.first_level], channels[self.first_level])
        # self.second_stage_dcn3 = DeformConv(channels[self.first_level], channels[self.first_level])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            in_channel = head_conv
            if head == 'proposal' or head == 'scale':
                in_channel = channels[self.first_level]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(in_channel, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                elif 'proposal' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(in_channel, classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                elif 'proposal' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)  # [1s, 2s, 4s, 8s, 16s, 32s]
        # base_feat = []
        # for i in x:
        #     base_feat.append(i.clone())

        dla_feat = self.dla_up(x)  # [4s, 8s, 16s, 32s]
        # for i in dla_feat:
        #     print(i.size())

        coarse_supervision_feat = []
        for i in range(self.last_level - self.first_level):
            coarse_supervision_feat.append(dla_feat[i].clone()) # [4s, 8s, 16s]
        self.ida_up(coarse_supervision_feat, 0, len(coarse_supervision_feat))

        out = {}
        fine_supervision_feat = coarse_supervision_feat[-1]
        if 'proposal' in self.heads:
            out['proposal'] = self.__getattr__('proposal')(coarse_supervision_feat[-1])
            out['scale'] = self.__getattr__('scale')(coarse_supervision_feat[-1])
            # fine_supervision_feat = fine_supervision_feat * self.sigmoid(out['proposal'])
            fine_supervision_feat = self.feature_adaptation(fine_supervision_feat, out['proposal'], out['scale'])

        second_stage_conv0 = self.second_stage_conv0(fine_supervision_feat)
        second_stage_conv1 = self.second_stage_conv1(second_stage_conv0)

        # method 1
        # second_stage_stride4 = self.second_stage_csa0(base_feat[2], dla_feat[0], coarse_supervision_feat[-1])
        # # second_stage_stride4 = dla_feat[0]
        # second_stage_stride8 = self.second_stage_bottleneck0(second_stage_stride4)
        # second_stage_stride8 = self.second_stage_csa1(base_feat[3], dla_feat[1], second_stage_stride8)
        # second_stage_stride16 = self.second_stage_bottleneck1(second_stage_stride8)
        # second_stage_stride16 = self.second_stage_csa2(base_feat[4], dla_feat[2], second_stage_stride16)
        # second_stage_stride32 = self.second_stage_bottleneck2(second_stage_stride16)
        # second_stage_stride32 = self.second_stage_csa3(base_feat[5], dla_feat[3], second_stage_stride32)
        # #
        # # second_stage_feat = self.second_stage_feature_fusion([second_stage_stride4, second_stage_stride8,
        # #                                                             second_stage_stride16, second_stage_stride32])
        # second_stage_feat = self.second_stage_dla_up([second_stage_stride4, second_stage_stride8,
        #                                               second_stage_stride16, second_stage_stride32])
        # fine_supervision_feat = []
        # for i in second_stage_feat:
        #     fine_supervision_feat.append(i.clone())  # [4s, 8s, 16s]
        #
        # self.second_stage_ida_up(fine_supervision_feat, 0, len(fine_supervision_feat))

        for head in self.heads:
            if head == 'proposal' or head == 'scale':
                continue
            out[head] = self.__getattr__(head)(second_stage_conv1)
            # z.append(self.__getattr__(head)(y[-1]))
        return [out]

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  # model = DLASeg('dla{}'.format(num_layers), heads,
  #                pretrained=True,
  #                down_ratio=down_ratio,
  #                final_kernel=1,
  #                last_level=5,
  #                head_conv=head_conv)
  model = TwoStageDLASeg('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model


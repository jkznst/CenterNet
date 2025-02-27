from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os
import tempfile

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.pose_dla_dcn import get_msp_pose_net as get_msp_dla_dcn
from .networks.pose_dla_dcn import get_two_stage_pose_net as get_two_stage_dla_dcn
from .networks.pose_dla_dcn import get_two_stage_msp_pose_net as get_two_stage_msp_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net
from utils.oss_tools import OSS_Bucket

_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'twostagedla': get_two_stage_dla_dcn,
  'resdcn': get_pose_net_dcn,
  'hourglass': get_large_hourglass_net,
}

_msp_model_factory = {
  'dla': get_msp_dla_dcn,
  'twostagedla': get_two_stage_msp_dla_dcn
}

def create_model(arch, heads, head_conv, msp=False):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  if not msp:
    get_model = _model_factory[arch]
  else:
    get_model = _msp_model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0

  if OSS_Bucket.oss_bucket:
    tmp = tempfile.NamedTemporaryFile()
    try:
      OSS_Bucket.bucket.get_object_to_file(model_path, tmp.name)
    except Exception as e:
      print(e)
    checkpoint = torch.load(tmp, map_location=lambda storage, loc: storage)
    tmp.close()
  else:
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}.'.format(
          k, model_state_dict[k].shape, state_dict[k].shape))
        state_dict[k] = model_state_dict[k]
        # tailor coco heatmap weight
        # state_dict[k] = state_dict[k][0:1]
        # assert state_dict[k].shape == model_state_dict[k].shape
        # print("use person heatmap weight!")
    else:
      print('Drop parameter {}.'.format(k))
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k))
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  # torch.save(data, path)

  if OSS_Bucket.oss_bucket:
    tmp = tempfile.NamedTemporaryFile()
    torch.save(data, tmp)
    try:
      OSS_Bucket.bucket.put_object_from_file(path, tmp.name)
    except Exception as e:
      print(e)
    tmp.close()
  else:
    torch.save(data, path)

def save_onnx_model(model, path="model.onnx"):
  dummy_input = torch.randn(1, 3, 288, 512, requires_grad=True)
  torch.onnx.export(model, dummy_input, path, verbose=True)


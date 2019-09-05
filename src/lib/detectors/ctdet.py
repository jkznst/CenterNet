from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from external.nms import soft_nms
from models.decode import ctdet_decode, ctdet_twostage_decode, ctdet_multiscale_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None
      scale = output['scale'] if self.opt.reg_proposal else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
        scale = (scale[0:1] + flip_tensor(scale[0:1])) / 2
      torch.cuda.synchronize()
      forward_time = time.time()
      # dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)
      dets = ctdet_twostage_decode(hm, wh, reg=reg, scale=scale, K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def multiscale_process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm_small = output['hm_small'].sigmoid_()
      hm_medium = output['hm_medium'].sigmoid_()
      hm_big = output['hm_big'].sigmoid_()
      wh_small = output['wh_small']
      wh_medium = output['wh_medium']
      wh_big = output['wh_big']
      reg_small = output['reg_small'] if self.opt.reg_offset else None
      reg_medium = output['reg_medium'] if self.opt.reg_offset else None
      reg_big = output['reg_big'] if self.opt.reg_offset else None
      scale = output['scale'] if self.opt.reg_proposal else None
      if self.opt.flip_test:
        hm_small = (hm_small[0:1] + flip_tensor(hm_small[1:2])) / 2
        wh_small = (wh_small[0:1] + flip_tensor(wh_small[1:2])) / 2
        reg_small = reg_small[0:1] if reg_small is not None else None
        hm_medium = (hm_medium[0:1] + flip_tensor(hm_medium[1:2])) / 2
        wh_medium = (wh_medium[0:1] + flip_tensor(wh_medium[1:2])) / 2
        reg_medium = reg_medium[0:1] if reg_medium is not None else None
        hm_big = (hm_big[0:1] + flip_tensor(hm_big[1:2])) / 2
        wh_big = (wh_big[0:1] + flip_tensor(wh_big[1:2])) / 2
        reg_big = reg_big[0:1] if reg_big is not None else None
        scale = (scale[0:1] + flip_tensor(scale[0:1])) / 2
      torch.cuda.synchronize()
      forward_time = time.time()
      # dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)
      dets = ctdet_multiscale_decode(hm_small, hm_medium, hm_big,
                                     wh_small, wh_medium, wh_big,
                                     reg_small, reg_medium, reg_big,
                                     scale=scale, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results, img_id=None):
    debugger.add_img(image, img_id='{:d}'.format(img_id.numpy()[0]))
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='{:d}'.format(img_id.numpy()[0]))
    debugger.save_img(imgId='{:d}'.format(img_id.numpy()[0]), path=self.opt.debug_dir)
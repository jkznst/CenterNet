from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_proposal_gaussian, draw_scale, draw_sfaf_proposal
from utils.image import draw_dense_reg
from utils.zipreader import ZipReader
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    # self.zip_reader = ZipReader(flags=cv2.IMREAD_COLOR, oss_bucket=self.opt.oss)
    if not self.opt.oss:
      img_path = os.path.join(self.img_dir, file_name)
      img = cv2.imread(img_path)
    else:
      img_path = os.path.join(self.img_dir, '@', file_name)
      self.zip_reader = ZipReader(flags=cv2.IMREAD_COLOR, oss_bucket=self.opt.oss)
      img = self.zip_reader.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm_small = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    hm_medium = np.zeros((num_classes, output_h // 2, output_w // 2), dtype=np.float32)
    hm_big = np.zeros((num_classes, output_h // 4, output_w // 4), dtype=np.float32)
    hm_mask = np.ones((1, output_h, output_w), dtype=np.float32)
    proposal = np.zeros((1, output_h, output_w), dtype=np.float32)
    proposal_scale = np.zeros((1, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask_small = np.zeros((self.max_objs), dtype=np.uint8)
    reg_mask_medium = np.zeros((self.max_objs), dtype=np.uint8)
    reg_mask_big = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    # if file_name.find("sur") == 0:
    #   for k in range(num_objs):
    #     ann = anns[k]
    #     bbox = self._coco_box_to_bbox(ann['bbox'])
    #     if ann['category_id'] > num_classes + 1:
    #       continue
    #     elif ann['category_id'] == num_classes + 1:
    #       # ignore class
    #       if flipped:
    #         bbox[[0, 2]] = width - bbox[[2, 0]] - 1
    #       bbox[:2] = affine_transform(bbox[:2], trans_output)
    #       bbox[2:] = affine_transform(bbox[2:], trans_output)
    #       bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
    #       bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
    #       h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    #       if h > 0 and w > 0:
    #         hm_mask[0, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0.0

    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      # if ann['category_id'] > num_classes:
      #   continue
      cls_id = int(self.cat_ids[ann['category_id']])

      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0 and np.sqrt(h * w) < 32:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        #print(h, w, radius)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if hm_mask[0, ct_int[1], ct_int[0]] > 0:
          draw_gaussian(hm_small[cls_id], ct_int, radius)
          # the same regions of adjcent levels are set as IR
          ct_medium = ct / 2.0
          ct_medium_int = ct_medium.astype(np.int32)
          radius_medium = gaussian_radius((math.ceil(h / 2.0), math.ceil(w / 2.0)))
          radius_medium = max(0, int(radius_medium))
          radius_medium = self.opt.hm_gauss if self.opt.mse_loss else radius_medium
          draw_gaussian(hm_medium[cls_id], ct_medium_int, radius_medium)
          hm_medium[cls_id, ct_medium_int[1], ct_medium_int[0]] = 0.9999
          # draw_proposal_gaussian(proposal[0], ct_int, int(h), int(w))
          # draw_gaussian(proposal[0], ct_int, 2 * (radius + 1))
          # scale = np.sqrt(h * w)
          # draw_scale(proposal_scale[0], bbox, scale)
          draw_sfaf_proposal(proposal[0], proposal_scale[0], ct_int, h, w)
          # draw_gaussian(proposal[cls_id], ct_int, radius)
          wh[k] = 1. * w, 1. * h
          ind[k] = ct_int[1] * output_w + ct_int[0]
          reg[k] = ct - ct_int
          reg_mask_small[k] = 1
          cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
          cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh, hm_small.max(axis=0), ct_int, wh[k], radius)
          gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                         ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
      elif 32 <= np.sqrt(h * w) < 64:
        h /= 2.0
        w /= 2.0
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        # print(h, w, radius)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct /= 2.0
        ct_int = ct.astype(np.int32)
        if hm_mask[0, ct_int[1] * 2, ct_int[0] * 2] > 0:
          draw_gaussian(hm_medium[cls_id], ct_int, radius)
          # the same regions of adjcent levels are set as IR
          ct_small = ct * 2.0
          ct_small_int = ct_small.astype(np.int32)
          radius_small = gaussian_radius((math.ceil(h * 2.0), math.ceil(w * 2.0)))
          radius_small = max(0, int(radius_small))
          radius_small = self.opt.hm_gauss if self.opt.mse_loss else radius_small
          draw_gaussian(hm_small[cls_id], ct_small_int, radius_small)
          hm_small[cls_id, ct_small_int[1], ct_small_int[0]] = 0.9999

          ct_big = ct / 2.0
          ct_big_int = ct_big.astype(np.int32)
          radius_big = gaussian_radius((math.ceil(h / 2.0), math.ceil(w / 2.0)))
          radius_big = max(0, int(radius_big))
          radius_big = self.opt.hm_gauss if self.opt.mse_loss else radius_big
          draw_gaussian(hm_big[cls_id], ct_big_int, radius_big)
          hm_big[cls_id, ct_big_int[1], ct_big_int[0]] = 0.9999
          # draw_proposal_gaussian(proposal[0], ct_int, int(h), int(w))
          # draw_gaussian(proposal[0], ct_int, 2 * (radius + 1))
          # scale = np.sqrt(h * w)
          # draw_scale(proposal_scale[0], bbox, scale)
          draw_sfaf_proposal(proposal[0], proposal_scale[0], ct_int, h, w)
          # draw_gaussian(proposal[cls_id], ct_int, radius)
          wh[k] = 1. * w, 1. * h
          ind[k] = ct_int[1] * output_w // 2 + ct_int[0]
          reg[k] = ct - ct_int
          reg_mask_medium[k] = 1
          cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
          cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh, hm_small.max(axis=0), ct_int, wh[k], radius)
          gt_det.append([2 * (ct[0] - w / 2), 2 * (ct[1] - h / 2),
                         2 * (ct[0] + w / 2), 2 * (ct[1] + h / 2), 1, cls_id])
      elif np.sqrt(h * w) >= 64:
        h /= 4.0
        w /= 4.0
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        # print(h, w, radius)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct /= 4.0
        ct_int = ct.astype(np.int32)
        if hm_mask[0, ct_int[1] * 4, ct_int[0] * 4] > 0:
          draw_gaussian(hm_big[cls_id], ct_int, radius)
          # the same regions of adjcent levels are set as IR
          ct_medium = ct * 2.0
          ct_medium_int = ct_medium.astype(np.int32)
          radius_medium = gaussian_radius((math.ceil(h * 2.0), math.ceil(w * 2.0)))
          radius_medium = max(0, int(radius_medium))
          radius_medium = self.opt.hm_gauss if self.opt.mse_loss else radius_medium
          draw_gaussian(hm_medium[cls_id], ct_medium_int, radius_medium)
          hm_medium[cls_id, ct_medium_int[1], ct_medium_int[0]] = 0.9999
          # draw_proposal_gaussian(proposal[0], ct_int, int(h), int(w))
          # draw_gaussian(proposal[0], ct_int, 2 * (radius + 1))
          # scale = np.sqrt(h * w)
          # draw_scale(proposal_scale[0], bbox, scale)
          draw_sfaf_proposal(proposal[0], proposal_scale[0], ct_int, h, w)
          # draw_gaussian(proposal[cls_id], ct_int, radius)
          wh[k] = 1. * w, 1. * h
          ind[k] = ct_int[1] * output_w // 4 + ct_int[0]
          reg[k] = ct - ct_int
          reg_mask_big[k] = 1
          cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
          cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh, hm_small.max(axis=0), ct_int, wh[k], radius)
          gt_det.append([4 * (ct[0] - w / 2), 4 * (ct[1] - h / 2),
                         4 * (ct[0] + w / 2), 4 * (ct[1] + h / 2), 1, cls_id])
    
    ret = {'input': inp,
           'hm_small': hm_small, 'hm_medium': hm_medium, 'hm_big': hm_big,
           'reg_mask_small': reg_mask_small, 'reg_mask_medium': reg_mask_medium, 'reg_mask_big': reg_mask_big,
           'ind': ind, 'wh': wh, 'hm_mask': hm_mask}
    if self.opt.dense_wh:
      hm_a = hm_small.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.reg_proposal:
      ret.update({'proposal': proposal})
      ret.update({'scale': proposal_scale})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret
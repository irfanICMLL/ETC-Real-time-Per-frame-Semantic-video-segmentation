##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## updated by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch.nn as nn
from torch.nn import functional as F
import torch
import os
import sys
import functools

affine_par = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
BatchNorm2d = functools.partial(nn.BatchNorm2d)


class Sd_model(nn.Module):
    def __init__(self, student, teacher, criterion, args):
        super(Sd_model, self).__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.args = args

    def inference(self, image):
        self.student.eval()
        _, _, w, h = image.size()
        preds = self.student(image)

        scale_pred = F.upsample(input=preds['logits'], size=(w, h), mode='bilinear', align_corners=True)

        return scale_pred

    def cal_loss(self, s_image, s_frames, label):
        self.teacher.eval()
        self.student.train()
        _, _, w, h = s_image.size()
        preds = self.student(s_image)
        preds_frame = self.student(s_frames)
        with torch.no_grad():
            soft = self.teacher(s_image)
            soft_frame = self.teacher(s_frames)
        loss = torch.zeros(1).cuda()
        loss_k = torch.zeros(1).cuda()
        loss_l = torch.zeros(1).cuda()
        loss_s = torch.zeros(1).cuda()
        if 'label' in self.args.sd_mode:
            loss_l = self.args.weight_l * self.criterion['ce'](preds, label)
            loss = loss + loss_l
        if 'kd' in self.args.sd_mode:
            loss_k = self.args.weight_k * self.criterion['kd'](preds_frame, soft_frame,
                                                               preds, soft)

            loss = loss + loss_k
        if 'sd' in self.args.sd_mode:
            loss_s = self.args.weight_s * self.criterion['sd'](preds_frame, soft_frame,
                                                               preds, soft)
            loss = loss + loss_s
        scale_pred = F.upsample(input=preds['logits'], size=(w, h), mode='bilinear', align_corners=True)
        output = scale_pred.max(1)[1]
        return {'output': output, 'total_loss': loss, 'loss_sd': loss_s, 'loss_kd': loss_k, 'loss_l': loss_l}

    def forward(self, s_image, s_frames, label, mode):
        if mode == 'train':
            ret = self.cal_loss(s_image, s_frames, label)
        elif mode == 'test':
            ret = self.inference(s_image)

        return ret

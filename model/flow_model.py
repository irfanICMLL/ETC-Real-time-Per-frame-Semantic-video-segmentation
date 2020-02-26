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
from flownet import *
from flownet.resample2d_package.resample2d import Resample2d

affine_par = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
BatchNorm2d = functools.partial(nn.BatchNorm2d)

TAG_CHAR = np.array([202021.25], np.float32)


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


class FlowModel(nn.Module):
    def __init__(self, student, criterion, args):
        super(FlowModel, self).__init__()
        self.student = student
        args.rgb_max = 255.0
        args.fp16 = False
        self.flownet = FlowNet2(args, requires_grad=False)
        checkpoint = torch.load(
            "./pretrained_model/FlowNet2_checkpoint.pth.tar")
        self.flownet.load_state_dict(checkpoint['state_dict'])
        self.flow_warp = Resample2d()
        self.criterion = criterion
        self.args = args
        self.criterion_flow = nn.MSELoss(size_average=True)

    def inference(self, image):
        self.student.eval()
        _, _, w, h = image.size()
        preds = self.student(image)

        scale_pred = F.upsample(input=preds['logits'], size=(w, h), mode='bilinear', align_corners=True)

        return scale_pred

    def cal_loss(self, s_image, s_frames, label):
        self.student.train()
        self.flownet.eval()

        preds = self.student(s_image)
        preds_frame = self.student(s_frames)

        _, _, w, h = s_image.size()
        scale_pred = F.upsample(input=preds['logits'], size=(w, h), mode='bilinear', align_corners=True)
        scale_frames = F.upsample(input=preds_frame['logits'], size=(w, h), mode='bilinear', align_corners=True)
        # prob = F.softmax(scale_pred, dim=1)
        # for j in range(scale_pred.shape[0]):
        #     for i in range(19):
        #         mask = prob[j][i]
        #         cv2.imwrite('../plot_main_fig/logits/%d_%d_pred.png' % (j, i),
        #                     mask.cpu().data.numpy() * 255)
        with torch.no_grad():
            flow_i21 = self.flownet(s_image, s_frames)
        # import cvbase as cvb
        # cvb.show_flow(flow_i21[0].cpu().data.numpy().transpose(1, 2, 0))

        warp_i1 = self.flow_warp(s_frames, flow_i21)
        warp_o1 = self.flow_warp(scale_frames, flow_i21)

        noc_mask2 = torch.exp(-1 * torch.abs(torch.sum(s_image - warp_i1, dim=1))).unsqueeze(1)
        ST_loss = self.args.st_weight * self.criterion_flow(scale_pred * noc_mask2, warp_o1 * noc_mask2)
        loss_ce = self.criterion['ce'](preds, label)
        output = scale_pred.max(1)[1]
        loss = {'output': output, 'ce_loss': loss_ce, 'st_loss': ST_loss, 'total_loss': ST_loss + loss_ce}
        return loss

    def forward(self, s_image, s_frames, label, mode):
        if mode == 'train':
            ret = self.cal_loss(s_image, s_frames, label)
        elif mode == 'test':
            ret = self.inference(s_image)

        return ret


if __name__ == '__main__':
    import os
    import time
    import argparse
    import cv2
    import numpy

    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--st_weight', type=float, default=0.4, help='st_weight')
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    mean = (0.485 * 255, 0.456 * 255, 0.406 * 255)
    std = (0.229 * 255, 0.224 * 255, 0.225 * 255)
    frame1 = cv2.imread('../plot_main_fig/stuttgart_00_000000_000001_leftImg8bit.png')
    frame2 = cv2.imread('../plot_main_fig/stuttgart_00_000000_000003_leftImg8bit.png')
    frame3 = cv2.imread('../plot_main_fig/stuttgart_00_000000_000005_leftImg8bit.png')
    input1 = torch.from_numpy(numpy.array((frame1 - mean) / std).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
    input2 = torch.from_numpy(numpy.array((frame2 - mean) / std).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
    input3 = torch.from_numpy(numpy.array((frame3 - mean) / std).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
    input_12 = torch.cat((input1, input2), dim=0)
    input_23 = torch.cat((input2, input3), dim=0)
    from model.pspnet_18 import *

    student = PSPNet(layers=18, bins=(1, 2, 3, 6), dropout=0.1, classes=19, zoom_factor=1, use_ppm=True,
                     pretrained=False, flow=True).cuda()
    ckpt = torch.load('../exp/cityscapes/pspnet18_base/model/eval_para.pth')
    student = torch.nn.DataParallel(student)
    student.load_state_dict(ckpt)
    args = parser.parse_args()
    criterion = torch.nn.CrossEntropyLoss()
    criterion_list = nn.ModuleDict({'ce': criterion})
    model = FlowModel(student, criterion_list, args)
    model.cuda()
    res = model(input_12, input_23, input_23, 'train')


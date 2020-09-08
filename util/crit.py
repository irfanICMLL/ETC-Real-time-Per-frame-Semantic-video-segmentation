import pdb
import torch.nn as nn
import math
import os
import sys
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from model.ConvLSTM import ConvLSTM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
torch_ver = torch.__version__[:3]


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, use_weight=True):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.use_weight = use_weight
        if self.use_weight:
            self.weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]).cuda()
            print('CrossEntropy2d weights : {}'.format(self.weight))
        else:
            self.weight = None

    def forward(self, predict, target, weight=None):

        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # Variable(torch.randn(2,10)
        if self.use_weight:
            print('target size {}'.format(target.shape))
            freq = np.zeros(19)
            for k in range(19):
                mask = (target[:, :, :] == k)
                freq[k] = torch.sum(mask)
                print('{}th frequency {}'.format(k, freq[k]))
            weight = freq / np.sum(freq)
            print(weight)
            self.weight = torch.FloatTensor(weight)
            print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = None

        criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_label)
        # torch.FloatTensor([2.87, 13.19, 5.11, 37.98, 35.14, 30.9, 26.23, 40.24, 6.66, 32.07, 21.08, 28.14, 46.01, 10.35, 44.25, 44.9, 44.25, 47.87, 40.39])
        # weight = Variable(torch.FloatTensor([1, 1.49, 1.28, 1.62, 1.62, 1.62, 1.64, 1.62, 1.49, 1.62, 1.43, 1.62, 1.64, 1.43, 1.64, 1.64, 1.64, 1.64, 1.62]), requires_grad=False).cuda()
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = criterion(predict, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight=True):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class CriterionCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
             0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds['logits'], size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, ignore_index=255, mode='logits', upsample=False, use_weight=True, T=1):
        super(CriterionKD, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.upsample = upsample
        self.T = T
        self.mode = mode
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds_frames, soft_frames, preds, soft):
        preds_frames = preds_frames[self.mode]
        soft_frame = soft_frames[self.mode]
        preds = preds[self.mode]
        soft = soft[self.mode]

        h, w = soft.size(2), soft.size(3)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        scale_frames = F.upsample(input=preds_frames, size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion_kd(F.log_softmax(scale_pred / self.T, dim=1), F.softmax(soft / self.T, dim=1))
        loss2 = self.criterion_kd(F.log_softmax(scale_frames / self.T, dim=1), F.softmax(soft_frame / self.T, dim=1))
        return loss1 + loss2


class CriterionKDLong(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, ignore_index=255, mode='logits', upsample=False, use_weight=True, T=1):
        super(CriterionKDLong, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.upsample = upsample
        self.T = T
        self.mode = mode
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds_frames_f, soft_frames_f, preds_frames_b, soft_frames_b, preds, soft):
        preds_frames_f = preds_frames_f[self.mode]
        soft_frame_f = soft_frames_f[self.mode]
        preds_frames_b = preds_frames_b[self.mode]
        soft_frame_b = soft_frames_b[self.mode]
        preds = preds[self.mode]
        soft = soft[self.mode]

        h, w = soft.size(2), soft.size(3)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        scale_frames_f = F.upsample(input=preds_frames_f, size=(h, w), mode='bilinear', align_corners=True)
        scale_frames_b = F.upsample(input=preds_frames_b, size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion_kd(F.log_softmax(scale_pred / self.T, dim=1), F.softmax(soft / self.T, dim=1))
        loss2 = self.criterion_kd(F.log_softmax(scale_frames_f / self.T, dim=1),
                                  F.softmax(soft_frame_f / self.T, dim=1))
        loss3 = self.criterion_kd(F.log_softmax(scale_frames_b / self.T, dim=1),
                                  F.softmax(soft_frame_b / self.T, dim=1))
        return loss1 + 0.4 * (loss2 + loss3)


class Cos_Attn_no(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_no, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, preds, preds_frames):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = preds.size()
        proj_query = preds.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = preds_frames.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        p_norm = proj_key.norm(2, dim=1)
        nm = torch.bmm(p_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        # attention = self.softmax(norm_energy)  # BX (N) X (N)
        return norm_energy


class CriterionSDcos_sig(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, mode='logits', use_weight=True):
        super(CriterionSDcos_sig, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.attn = Cos_Attn_no('relu')
        self.criterion_sd = torch.nn.MSELoss()
        self.mode = mode

    def forward(self, preds_frames, soft_frames, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        preds_frames = preds_frames[self.mode]
        soft_frame = soft_frames[self.mode]
        preds = preds[self.mode]
        soft = soft[self.mode]
        graph_s = self.attn(preds, preds_frames)
        graph_t = self.attn(soft, soft_frame)
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionSDcos_self(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, mode='fea', use_weight=True, pool=2):
        super(CriterionSDcos_self, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.pool = pool
        self.mode = mode
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn_self('relu')
        self.criterion_sd = torch.nn.MSELoss()
        self.avgpool = nn.AvgPool2d(kernel_size=pool, stride=2)

    def forward(self, preds_frames_f, soft_frames_f, preds_frames_b, soft_frames_b, preds, soft):
        preds_frames_f = preds_frames_f[self.mode]
        soft_frame_f = soft_frames_f[self.mode]
        preds_frames_b = preds_frames_b[self.mode]
        soft_frame_b = soft_frames_b[self.mode]
        preds = preds[self.mode]
        soft = soft[self.mode]

        pool_soft = self.avgpool(soft)
        pool_preds = self.avgpool(preds)

        pool_soft_frame_b = self.avgpool(soft_frame_b)
        pool_preds_frame_b = self.avgpool(preds_frames_b)

        pool_soft_frame_f = self.avgpool(soft_frame_f)
        pool_preds_frame_f = self.avgpool(preds_frames_f)

        graph_s = self.attn(pool_soft)
        graph_p = self.attn(pool_preds)
        graph_pf = self.attn(pool_preds_frame_f)
        graph_sf = self.attn(pool_soft_frame_f)
        graph_pb = self.attn(pool_preds_frame_b)
        graph_sb = self.attn(pool_soft_frame_b)
        loss0 = self.criterion_sd(graph_p, graph_s)
        loss1 = self.criterion_sd(graph_pf, graph_sf)
        loss2 = self.criterion_sd(graph_pb, graph_sb)

        return loss0 + 0.4 * (loss1 + loss2)


class CriterionLSTMGAN(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, mode='fea', use_weight=True, pool=2):
        super(CriterionLSTMGAN, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.pool = pool
        self.mode = mode
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn_self('relu')
        self.criterion_sd = torch.nn.MSELoss()
        nf = 1
        self.convlstm = ConvLSTM(input_size=1, hidden_size=nf * 4, kernel_size=3)

    def forward(self, preds_frames_f, soft_frames_f, preds_frames_b, soft_frames_b, preds, soft):
        s_seq = []
        t_seq = []
        preds_frames_f = preds_frames_f[self.mode]
        soft_frame_f = soft_frames_f[self.mode]
        preds_frames_b = preds_frames_b[self.mode]
        soft_frame_b = soft_frames_b[self.mode]
        preds = preds[self.mode]
        soft = soft[self.mode]
        t_seq = [soft_frame_f, soft, soft_frame_b]
        s_seq = [preds_frames_f, preds, preds_frames_b]

        state = None
        for i in s_seq:
            n, c, w, h = i.shape
            i = torch.reshape(i, (-1, w, h))
            i = i.unsqueeze(1)
            state = self.convlstm(i, state)
        state_t = None
        for i in t_seq:
            n, c, w, h = i.shape
            i = torch.reshape(i, (-1, w, h))
            i = i.unsqueeze(1)
            state_t = self.convlstm(i, state_t)
        loss1 = self.criterion_sd(state_t[1], state[1])
        lstm_tea = torch.mean(torch.ones(state_t[1].size()).cuda() - state_t[1])
        return loss1, lstm_tea


class CriterionLSTM(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, mode='fea', use_weight=True, pool=2):
        super(CriterionLSTM, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.pool = pool
        self.mode = mode
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn_self('relu')
        self.criterion_sd = torch.nn.MSELoss()
        self.avgpool = nn.AvgPool2d(kernel_size=16, stride=16)
        nf = 1
        self.convlstm = ConvLSTM(input_size=1, hidden_size=nf * 4, kernel_size=3)

    def forward(self, preds_frames_f, soft_frames_f, preds_frames_b, soft_frames_b, preds, soft):
        s_seq = []
        t_seq = []
        preds_frames_f = preds_frames_f[self.mode]
        soft_frame_f = soft_frames_f[self.mode]
        preds_frames_b = preds_frames_b[self.mode]
        soft_frame_b = soft_frames_b[self.mode]
        preds = preds[self.mode]
        soft = soft[self.mode]

        pool_soft = self.avgpool(soft)
        pool_preds = self.avgpool(preds)

        pool_soft_frame_b = self.avgpool(soft_frame_b)
        pool_preds_frame_b = self.avgpool(preds_frames_b)

        pool_soft_frame_f = self.avgpool(soft_frame_f)
        pool_preds_frame_f = self.avgpool(preds_frames_f)
        graph_sf = self.attn(pool_soft_frame_f)
        t_seq.append(graph_sf)
        graph_s = self.attn(pool_soft)
        t_seq.append(graph_s)
        graph_sb = self.attn(pool_soft_frame_b)
        t_seq.append(graph_sb)

        graph_pf = self.attn(pool_preds_frame_f)
        s_seq.append(graph_pf)
        graph_p = self.attn(pool_preds)
        s_seq.append(graph_p)
        graph_pb = self.attn(pool_preds_frame_b)
        s_seq.append(graph_pb)
        state = None
        for i in s_seq:
            i = i.unsqueeze(1)
            state = self.convlstm(i, state)
        state_t = None
        for i in t_seq:
            i = i.unsqueeze(1)
            state_t = self.convlstm(i, state_t)
        loss1 = self.criterion_sd(state_t[1], state[1])
        loss0 = self.criterion_sd(state_t[0], state[0])

        return loss1 + loss0


class Cos_Attn_self(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_self, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        # attention = self.softmax(norm_energy)  # BX (N) X (N)
        return norm_energy


class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor(
            [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
             0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds['dsn'], size=(h, w), mode='bilinear', align_corners=True)

        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds['logits'], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionOhemDSN(nn.Module):
    '''
    DSN + OHEM : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4, use_weight=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds['dsn'], size=(h, w), mode='bilinear', align_corners=True)

        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.upsample(input=preds['logits'], size=(h, w), mode='bilinear', align_corners=True)

        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionOhemDSN_single(nn.Module):
    '''
    DSN + OHEM : we find that use hard-mining for both supervision harms the performance.
                Thus we choose the original loss for the shallow supervision
                and the hard-mining loss for the deeper supervision
    '''

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4):
        super(CriterionOhemDSN_single, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor(
            [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
             0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=True)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds['dsn'], size=(h, w), mode='bilinear', align_corners=True)

        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.upsample(input=preds['logits'], size=(h, w), mode='bilinear', align_corners=True)

        loss2 = self.criterion_ohem(scale_pred, target)
        return self.dsn_weight * loss1 + loss2

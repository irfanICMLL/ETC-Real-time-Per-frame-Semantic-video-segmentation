import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import sys
from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cv2.ocl.setUseOpenCL(False)
from flownet import *
from flownet.resample2d_package.resample2d import Resample2d

FLO_TAG = 202021.25


def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


def save_flo(flow, filename):
    flow = tensor2img(flow)
    with open(filename, 'wb') as f:
        tag = np.array([FLO_TAG], dtype=np.float32)

        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        tag.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/cityscapes/cityscapes_pspnet18_sd.yaml',
                        help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def read_flo(filename):
    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)

        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' % filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            # print 'Reading %d x %d flo file' % (w, h)

            data = np.fromfile(f, np.float32, count=2 * w * h)

            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))

    return flow


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if 'psp' in args.arch:
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (
                    args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    global args, logger
    args = get_parser()
    # check(args)
    logger = get_logger()
    list_of_list = open('data/list/cityscapes/val_video_list_sam.lst')
    args.rgb_max = 255.0
    args.fp16 = False
    flownet = FlowNet2(args, requires_grad=False).cuda()
    checkpoint = torch.load(
        "./pretrained_model/FlowNet2_checkpoint.pth.tar")
    flownet.load_state_dict(checkpoint['state_dict'])
    flow_warp = Resample2d().cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    # datalist = []
    # for j in data_list_folder:
    #     datalist.append(j)

    # if accel:
    #     gray_folder = os.path.join('/fast/users/a1760953/code/ongoing/Accel/demo/test_names/')
    # else:
    gray_folder = os.path.join(args.save_folder.replace('ss', 'video'), 'gray')

    cal_acc(list_of_list=list_of_list, root_dir='data/cityscapes',
            pred_folder=gray_folder,
            classes=19, flow_warp=flow_warp
            , flownet=flownet, names=names)


def normalize(image):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image / value_scale
    image = image - mean
    image = image / std
    return image


def color2grey(rgb_label, colors):
    grey_label = np.ones((1024, 2048)) * 255
    count_class = 0
    for l in colors:
        mask = rgb_label == l
        mask_single = mask.sum(axis=2)
        grey_label[mask_single == 3] = count_class
        count_class += 1
    return grey_label


def cal_acc(list_of_list, root_dir, pred_folder, classes, flow_warp, flownet, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for i in list_of_list:
        list_tem = './data/list/cityscapes/val_sam/' + i
        data_list = open(list_tem.strip())
        for i, image_path in enumerate(data_list):
            if i > 28:
                print('done!')
            else:
                image_name = image_path.split('/')[-1].split('.')[0]
                splits = image_path.split('_')
                frame_next = "_".join(
                    splits[:-2] + [(str(int(splits[-2]) + 1)).rjust(6, "0")] + splits[-1:])
                # if accel:
                #     bgr_label = cv2.imread(os.path.join(pred_folder, image_name + '.png'))
                #     rgb_label = cv2.cvtColor(bgr_label, cv2.COLOR_BGR2RGB)
                #     pred = color2grey(rgb_label, colors)
                # else:
                pred = cv2.imread(os.path.join(pred_folder, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
                print(os.path.join(pred_folder, image_name + '.png'))
                frame_name_next = frame_next.split('/')[-1].split('.')[0]
                pred_next = cv2.imread(os.path.join(pred_folder, frame_name_next + '.png'), cv2.IMREAD_GRAYSCALE)

                image = cv2.imread(os.path.join(root_dir, image_path.strip()),
                                   cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
                image = np.float32(image)
                image = normalize(image)
                frame = cv2.imread(os.path.join(root_dir, frame_next.strip()), cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
                frame = np.float32(frame)
                frame = normalize(frame)
                image_ten = torch.from_numpy(frame.astype(np.float32).transpose(2, 0, 1)).cuda().unsqueeze(0).cuda()
                frame_ten = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)).cuda().unsqueeze(0).cuda()
                pred_next_ten = torch.from_numpy(pred_next.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0).cuda()
                flownet.eval()

                output_flow_filename = os.path.join(root_dir + '/flow_val/', image_name + '.flo')
                if not os.path.exists(root_dir + '/flow_val/'):
                    os.makedirs(root_dir + '/flow_val/')
                if not os.path.exists(output_flow_filename):
                    print('lack of flow', output_flow_filename)
                    with torch.no_grad():
                        flow = flownet(frame_ten, image_ten)
                    save_flo(flow, output_flow_filename)
                else:
                    flow = read_flo(output_flow_filename)
                    flow = torch.from_numpy(flow.astype(np.float32).transpose(2, 0, 1)).cuda().unsqueeze(0).cuda()
                # warp_i1 = flow_warp(frame_ten, flow)
                warp_pred = flow_warp(pred_next_ten, flow)

                # noc_mask2 = torch.exp(-1 * torch.abs(torch.sum(image_ten - warp_i1, dim=1))).unsqueeze(1)
                # error=(warp_pred-pred)*noc_mask2

                intersection, union, target = intersectionAndUnion(pred,
                                                                   warp_pred[0][0].cpu().data.numpy().astype(np.uint8),
                                                                   classes)
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
                logger.info(
                    'Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, 29, image_name + '.png',
                                                                                accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i],
                                                                                    names[i]))


if __name__ == '__main__':
    main()


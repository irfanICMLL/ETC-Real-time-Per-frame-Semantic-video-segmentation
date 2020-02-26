import numpy as np
import cv2
import os
#
# folder_1 = '/hardware/yifanliu/sd_video_psp/exp/cityscapes/demo/color/'
# folder_2 = '/hardware/yifanliu/sd_video_psp/exp/cityscapes/demo_video/color/'
# # folder_1 = '/hardware/yifanliu/sd_video_psp/exp/cityscapes/val_demo/all/video/color/'
# # folder_2 = '/hardware/yifanliu/sd_video_psp/exp/cityscapes/val_demo/base/video/color/'
# list_1 = os.listdir(folder_1)
# list_2 = os.listdir(folder_2)
# list_1.sort()
# list_2.sort()
#
# k = 0.5
# c = 2
# b = 0
# x = np.tile(np.arange(0, 2048, 1), (1024, 1))
# y = np.tile(np.arange(0, 1024, 1), (2048, 1)).transpose(1, 0)
# up = 0
# r = 512
# step = 8
# output_dir = '/hardware/yifanliu/sd_video_psp/exp/cityscapes/seq00_demo_tri/'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# for i in range(len(list_1)):
#     dir1 = list_1[i]
#     dir2 = list_2[i]
#     image1 = cv2.imread(folder_1 + dir1)
#     image2 = cv2.imread(folder_2 + dir2)
#     if i >=0:
#         mask = y > k * x + b
#     else:
#         mask = x > 1024 + b
#     mask3 = np.tile(mask, (3, 1, 1)).transpose(1, 2, 0)
#     image1[mask3] = image2[mask3]
#     if i >=0:
#         mask_line1 = y > k * x + b - c
#         mask_line2 = y < k * x + b + c
#     else:
#         mask_line1 = x < 1026 + b
#         mask_line2 = x > 1022 + b
#
#     mask_line = mask_line1 & mask_line2
#     image1[mask_line] = 255
#
#     cv2.imwrite(output_dir + dir1, image1)
#     # if i > 0 and i < 2000:
#     #     if b < r and up == 1:
#     #         b = b + step
#     #     elif b >= r and up == 1:
#     #         b = r - 1
#     #         up = 0
#     #     elif b > -r and up == 0:
#     #         b = b - step
#     #     elif b <= -r and up == 0:
#     #         b = -r + 1
#     #         up = 1
#     # elif i >= 2000:
#     #     b = 0
#     print('b:', b, 'image:', dir1)
# # %%%%%%%%%%%%%%%%%%transfer color to grey%%%%%%%%%%%%%%
# folder = '/hardware/yifanliu/sd_video_psp/exp/testannot/'
# list_1 = os.listdir(folder)
# colors = np.loadtxt('/hardware/yifanliu/sd_video_psp/dataset/camvid/camvid_colors.txt')
# for i in list_1:
#     bgr_label = cv2.imread(folder + i)
#     rgb_label = cv2.cvtColor(bgr_label, cv2.COLOR_BGR2RGB)
#     grey_label = np.zeros((720, 960))
#     for j in range(11):
#         mask = rgb_label == colors[j]
#         mask_single = mask.sum(axis=2)
#         grey_label[mask_single == 3] = j
#     cv2.imwrite(folder+i.replace('_L.png','.png'),grey_label)
#     print(i)
# %%%%%%%%%%%%%%%%%%transfer color to grey%%%%%%%%%%%%%%
folder = '/hardware/yifanliu/sd_video_psp/data/camvid/testannot_L/'
list_1 = os.listdir(folder)
colors = np.loadtxt('/hardware/yifanliu/sd_video_psp/dataset/camvid/camvid_colors.txt')
origin_color_id = {(128, 0, 0): 0, (128, 128, 0): 1, (128, 128, 128): 2, (64, 0, 128): 3, (192, 128, 128): 4,
                   (128, 64, 128): 5, (64, 64, 0): 6, (64, 64, 128): 7, (192, 192, 128): 8, (0, 0, 192): 9,
                   (0, 128, 192): 10}

aug_color_id = {(128, 0, 0): 0, (128, 128, 0): 1, (128, 128, 128): 2, (64, 0, 128): 3, (192, 128, 128): 4,
                (128, 64, 128): 5, (64, 64, 0): 6, (64, 64, 128): 7, (192, 192, 128): 8, (0, 0, 192): 9,
                (0, 128, 192): 10, (192, 0, 192): 10, (192, 64, 128): 3, (192, 128, 192): 3, (64, 128, 192): 3,
                (64, 192, 0): 7, (192, 128, 64): 6, (128, 128, 192): 5, (64, 192, 128): 5, (192, 0, 128): 5,
                (128, 0, 192): 5, (192, 0, 64): 5, (128, 128, 64): 4, (192, 192, 0): 1}

for i in list_1:
    bgr_label = cv2.imread(folder + i)
    rgb_label = cv2.cvtColor(bgr_label, cv2.COLOR_BGR2RGB)
    grey_label = np.ones((720, 960)) * 11
    for l in origin_color_id:
        mask = rgb_label == l
        mask_single = mask.sum(axis=2)
        grey_label[mask_single == 3] = aug_color_id[l]
    cv2.imwrite(folder.replace('annot_L', 'annot_ori') + i.replace('_L.png', '.png'), np.uint8(grey_label))
    print(i)

import os
import numpy as np

video_list = '/media/data/yifan/code/sd_video_psp/data/list/cityscapes/val_sam/'
list_dir = '/media/data/yifan/code/sd_video_psp/data/list/cityscapes/val.lst'
list_dir_video = '/media/data/yifan/code/sd_video_psp/data/list/cityscapes/val_video_list_sam.lst'
img_dir_video = '/media/data/yifan/code/sd_video_psp/data/list/cityscapes/val_video_img_sam.lst'
f = open(list_dir)
list_of_list = open(list_dir_video, 'w')
all_img = open(img_dir_video, 'w')
for line in f:
    flag = np.random.random(1)
    if flag < 0.8:
        continue
    else:
        frame = []
        img, label = line.strip().split('\t')
        for i in range(19):
            splits = img.split('_')
            im_name = "_".join(
                splits[:-2] + [(str(int(splits[-2]) - (i + 1) * 1)).rjust(6, "0")] + splits[-1:])
            # print(im_name)
            frame.append(im_name)
        frame = frame[::-1]
        frame.append(img)
        for i in range(10):
            splits = img.split('_')
            im_name = "_".join(
                splits[:-2] + [(str(int(splits[-2]) + (i + 1) * 1)).rjust(6, "0")] + splits[-1:])
            # print(im_name)
            frame.append(im_name)
            # frame_t = cv2.imread(im_name, cv2.IMREAD_COLOR)
            # frames.append(frame_t)
        f_tem = open(video_list + img.split('/')[-1].replace('png', 'txt'), 'w')
        for i in frame:
            f_tem.writelines(i + '\n')
        for i in frame:
            all_img.writelines(i + '\n')
    f_tem.close()
    list_of_list.writelines(img.split('/')[-1].replace('png', 'txt') + '\n')
all_img.close()
list_of_list.close()
f.close()
# print(frame)
# print(line)

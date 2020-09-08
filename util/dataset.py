import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line_split = line.strip().split()
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, ignore_label=255):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.data_list)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label = self.id2trainId(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

class BaseCamvid(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, ignore_label=255, frame_gap=1,
                 random_frame=False):
        self.split = split
        image_label_list = []
        list_read = open(data_list).readlines()
        for line in list_read:
            line_split = line.strip().split()
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
            frame_name = os.path.join(data_root, line_split[2])
            item = (image_name, label_name, frame_name)
            image_label_list.append(item)
        self.data_list = image_label_list
        self.transform = transform
        self.frame_gap = frame_gap
        self.random_frame = random_frame
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.data_list)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        image_path, label_path, frame_path = self.data_list[index]
        image = cv2.imread(image_path.strip(), cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order

        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        # label = self.id2trainId(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, _, _, label = self.transform(image, image, image, label)
        return image, _, _, label


class VideoLongCamvid(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, ignore_label=255, frame_gap=1,
                 random_frame=False):
        self.split = split
        image_label_list = []
        list_read = open(data_list).readlines()
        for line in list_read:
            line_split = line.strip().split()
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
            frame_name = os.path.join(data_root, line_split[2])
            item = (image_name, label_name, frame_name)
            image_label_list.append(item)
        self.data_list = image_label_list
        self.transform = transform
        self.frame_gap = frame_gap
        self.random_frame = random_frame
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.data_list)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        image_path, label_path, frame_path = self.data_list[index]
        splits = frame_path.split('_')[-1].split('.')
        dir_keep = frame_path.split('_')[:-1]
        index = 0
        if self.random_frame:
            while index == 0:
                index = np.random.randint(low=0, high=self.frame_gap + 1, size=1)
            im_name_f = "_".join(
                dir_keep + [".".join([(str(int(splits[0]) - index[0])).rjust(5, "0")] + splits[-1:])])
            im_name_b = "_".join(
                dir_keep + [".".join([(str(int(splits[0]) + index[0])).rjust(5, "0")] + splits[-1:])])

        else:
            im_name_f = "_".join(
                dir_keep + [".".join([(str(int(splits[0]) - self.frame_gap)).rjust(5, "0")] + splits[-1:])])
            im_name_b = "_".join(
                dir_keep + [".".join([(str(int(splits[0]) + self.frame_gap)).rjust(5, "0")] + splits[-1:])])

        frame_f = cv2.imread(im_name_f.strip(), cv2.IMREAD_COLOR)
        frame_b = cv2.imread(im_name_b.strip(), cv2.IMREAD_COLOR)
        image = cv2.imread(image_path.strip(),
                           cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        try:
            image.shape
        except:
            print('fail to read', image_path)
        try:
            frame_f.shape
        except:
            print('fail to read', im_name_f)
        try:
            frame_b.shape
        except:
            print('fail to read', im_name_b)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        frame_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
        frame_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        frame_f = np.float32(frame_f)
        frame_b = np.float32(frame_b)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        # label = self.id2trainId(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, frame_f, frame_b, label = self.transform(image, frame_f, frame_b, label)
        return image, frame_f, frame_b, label


class VideoLongData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, ignore_label=255, frame_gap=1,
                 random_frame=False):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.frame_gap = frame_gap
        self.random_frame = random_frame
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.data_list)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        splits = image_path.split('_')
        index = 0
        if self.random_frame:
            while index == 0:
                index = np.random.randint(low=1, high=self.frame_gap, size=1)
            im_name_f = "_".join(
                splits[:-2] + [(str(int(splits[-2]) - index[0])).rjust(6, "0")] + splits[-1:])
            im_name_b = "_".join(
                splits[:-2] + [(str(int(splits[-2]) + index[0])).rjust(6, "0")] + splits[-1:])

        else:
            im_name_f = "_".join(
                splits[:-2] + [(str(int(splits[-2]) - self.frame_gap)).rjust(6, "0")] + splits[-1:])
            im_name_b = "_".join(
                splits[:-2] + [(str(int(splits[-2]) + self.frame_gap)).rjust(6, "0")] + splits[-1:])

        frame_f = cv2.imread(im_name_f, cv2.IMREAD_COLOR)
        frame_b = cv2.imread(im_name_b, cv2.IMREAD_COLOR)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        frame_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        frame_f = np.float32(frame_f)
        frame_b = np.float32(frame_b)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label = self.id2trainId(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, frame_f, frame_b, label = self.transform(image, frame_f, frame_b, label)
        return image, frame_f, frame_b, label


class VideoData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, ignore_label=255, frame_gap=1,
                 random_frame=False):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.frame_gap = frame_gap
        self.random_frame = random_frame
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.data_list)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        splits = image_path.split('_')
        index = 0
        if self.random_frame:
            while index == 0:
                index = np.random.randint(low=-self.frame_gap, high=self.frame_gap, size=1)
            im_name = "_".join(
                splits[:-2] + [(str(int(splits[-2]) - index[0])).rjust(6, "0")] + splits[-1:])
        else:
            im_name = "_".join(
                splits[:-2] + [(str(int(splits[-2]) - self.frame_gap)).rjust(6, "0")] + splits[-1:])
        frame = cv2.imread(im_name, cv2.IMREAD_COLOR)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        frame = np.float32(frame)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label = self.id2trainId(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, frame, label = self.transform(image, frame, label)
        return image, frame, label
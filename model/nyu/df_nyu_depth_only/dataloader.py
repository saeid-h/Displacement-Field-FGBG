import cv2
import numpy as np

import torch
from torch.utils import data

from config import config
from datareader.img_utils import random_scale, random_mirror, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape, \
    random_uniform_gaussian_blur, normalize_depth, rgb2gray

class TrainPre(object):
    def __init__(self, img_mean, img_std, target_size, use_gauss_blur=False):
        self.img_mean = img_mean
        self.img_std = img_std
        self.target_size = target_size
        self.use_gauss_blur = use_gauss_blur

    def __call__(self, img, gt, depth=None):
        img, gt, depth = random_mirror(img, gt, depth)
        if config.train_scale_array is not None:
            img, gt, depth, scale = random_scale(img, gt, depth, config.train_scale_array)

        #img = normalize(img, self.img_mean, self.img_std)
        img = rgb2gray(img)
        img = img / 255.

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 0)

        p_mask = np.zeros(p_gt.shape)
        p_mask[p_gt > 0] = 1
        if depth is None:
            p_depth = p_gt.copy()
        else:
            p_depth, _ = random_crop_pad_to_shape(depth, crop_pos, crop_size, 0)
        if self.use_gauss_blur:
            p_depth = random_uniform_gaussian_blur(p_depth, config.gaussian_kernel_range, config.max_kernel)
        p_gt = normalize_depth(p_gt)
        p_depth = normalize_depth(p_depth)

        p_img = np.expand_dims(p_img, axis=0)
        p_depth = np.expand_dims(p_depth, axis=0)
        extra_dict = None

        return p_img, p_depth, p_gt, p_mask, extra_dict

class TestPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std
        # self.target_size = target_size
        # self.use_gauss_blur = use_gauss_blur

    def __call__(self, img, gt, depth=None):
        img = rgb2gray(img)
        img = img / 255.
        p_mask = np.zeros(gt.shape)
        p_mask[gt > 0] = 1
        if depth is None: p_depth = gt.copy()
        # p_depth = normalize_depth(depth)
        depth = np.expand_dims(depth, axis=0)
        # p_gt = normalize_depth(gt)
        p_img = np.expand_dims(img, axis=0)
        extra_dict = None
        return p_img, depth, gt, p_mask, extra_dict

def get_train_loader(engine, dataset, filename_list=None):
    data_setting = {'data_source': config.data_source,
                    'train_test_splits': config.train_test_splits}
    train_preprocess = TrainPre(config.image_mean, config.image_std,
                                config.target_size, config.use_gauss_blur
                                )

    train_dataset = dataset(data_setting, "train", train_preprocess,
                            config.niters_per_epoch * config.batch_size,
                            filename_list=filename_list)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   shuffle=is_shuffle)

    return train_loader, train_sampler


def get_test_loader(engine, dataset, filename_list=None):
    data_setting = {'data_source': config.data_source,
                    'train_test_splits': config.train_test_splits}
    test_preprocess = TestPre(config.image_mean, config.image_std)
    test_dataset = dataset(data_setting, "test", test_preprocess, filename_list=filename_list)
    test_loader = data.DataLoader(test_dataset,
                                   batch_size=1,
                                   num_workers=config.num_workers,
                                   shuffle=False)
    return test_loader, test_dataset

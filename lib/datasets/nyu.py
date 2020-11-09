#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : BaseDataset.py

import os
import time
import cv2
import torch
import numpy as np
import os.path as osp
from PIL import Image

import torch.utils.data as data

class NYUDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, filename_list=None):
        super(NYUDataset, self).__init__()
        self._split_name = split_name
        self._file_length = None
        self.preprocess = preprocess
        self._filename_list = filename_list
        if self._filename_list is not None:
            with open(filename_list, 'r') as f:
                lines = f.readlines()
            if split_name == 'train':
                self.img_list = [osp.join('../../../dataset/sync', x.split(' ')[0]) for x in lines]
                self.gt_list = [osp.join('../../../dataset/sync', x.split(' ')[1]) for x in lines]
                self.depth_list = [osp.join('../../../dataset/nyu/bts_depth', 
                                x.split(' ')[0].replace(osp.sep,'_').replace('.jpg','.png')) for x in lines]
            else:
                self.img_list = [osp.join('../../../dataset/nyu', split_name, x.split(' ')[0]) for x in lines]
                self.gt_list = [osp.join('../../../dataset/nyu', split_name, x.split(' ')[1]) for x in lines]
            
            self.img_list.sort()
            self.gt_list.sort()
            self.depth_list.sort()
        else:
            self._split_name = split_name
            self._imgs_source, self._labels_source = self._load_data_file(setting['data_source'])
            self._train_test_splits = self._load_split_file(setting['train_test_splits'])
            self._file_names = self._get_file_names(split_name, self._train_test_splits)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        if self._filename_list is not None:
            return len(self.img_list)
        return len(self._file_names)

    def __getitem__(self, index):
        if self._filename_list is None:
            if self._file_length is not None:
                index_ = self._construct_new_file_names(self._file_length)[index]
            else:
                index_ = self._file_names[index]

            img, gt = self._fetch_data(index_)
            #img = img[:, :, ::-1]
            if self.preprocess is not None:
                img, ori, gt, mask, extra_dict = self.preprocess(img, gt)

            # if self._split_name is 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
            ori = torch.from_numpy(np.ascontiguousarray(ori)).float()
            mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].float()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()
            # else:
            #     img = torch.from_numpy(np.ascontiguousarray(img)).float()
            #     gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
            #     ori = torch.from_numpy(np.ascontiguousarray(ori)).float()
            #     mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

            output_dict = dict(guidance=img, data=ori, label=gt, mask=mask, n=len(self._file_names))
            if self.preprocess is not None and extra_dict is not None:
                output_dict.update(**extra_dict)

        else:
            img, gt, depth = self._fetch_data(index)     
            if self.preprocess is not None:
                img, depth, gt, mask, extra_dict = self.preprocess(img, gt, depth)

            if self._split_name is 'train':
                img = torch.from_numpy(np.ascontiguousarray(img)).float()
                gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
                depth = torch.from_numpy(np.ascontiguousarray(depth)).float()
                mask = torch.from_numpy(np.ascontiguousarray(mask)).float()       
            output_dict = dict(guidance=img, data=depth, label=gt, mask=mask, n=len(self.img_list))

        return output_dict


    def _fetch_data(self, index):
        if self._filename_list is None:
            img = self._imgs_source[index]
            gt = self._labels_source[index]
            return img, gt
        else:
            img = np.array(Image.open(self.img_list[index])).astype('float32')
            gt = np.array(Image.open(self.gt_list[index])).astype('float32') * 1000 / 65535
            depth = np.array(Image.open(self.depth_list[index])).astype('float32') * 1000 / 65535
            # depth = depth_read(self.depth_list[index])
            return img, gt, depth
        

    def _load_split_file(self, split_file_path):
        from scipy.io import loadmat
        split_file = loadmat(split_file_path)
        split_file['train'] = [i[0] - 1 for i in split_file['trainNdxs']]
        split_file['test'] = [i[0] - 1 for i in split_file['testNdxs']]
        return split_file

    def _load_data_file(self, data_source_path):
        import h5py
        data_file = h5py.File(data_source_path, 'r')
        depths = np.array(data_file['depths'], dtype=np.float64).transpose(0, 2, 1)
        images = np.array(data_file['images'], dtype=np.float64).transpose(0, 3, 2, 1)
        return images, depths


    def _get_file_names(self, split_name, split_file):
        assert split_name in ['train', 'test']
        file_names = split_file[split_name]

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()



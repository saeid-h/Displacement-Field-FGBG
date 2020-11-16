#!/usr/bin/env python3

import os
import time
import cv2
import torch
import numpy as np
import os.path as osp
from PIL import Image

import torch.utils.data as data

class ReplicaDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, filename_list=None):
        super(ReplicaDataset, self).__init__()
        self._split_name = split_name
        self.preprocess = preprocess
        self._filename_list = filename_list
        self.TAG_FLOAT = 202021.25
        self.TAG_CHAR = 'PIEH'
        with open(filename_list, 'r') as f:
            lines = f.readlines()
        self.img_list = [osp.join('../../../dataset/replica/image_left', x.split()[0]) for x in lines]
        self.gt_list = [osp.join('../../../dataset/replica/depth_left', x.split()[1]) for x in lines]
        if split_name == 'train':
            self.depth_list = [osp.join('../../../dataset/replica_bts_depth_train/raw', 
                            x.split(' ')[0].replace(osp.sep,'_').replace('.png','.dpt')) for x in lines]
        else:
            self.depth_list = [osp.join('../../../dataset/replica_bts_depth_test/raw', 
                            x.split(' ')[0].replace(osp.sep,'_').replace('.png','.dpt')) for x in lines]
        self.img_list.sort()
        self.gt_list.sort()
        self.depth_list.sort()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
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
        img = np.array(Image.open(self.img_list[index])).astype('float32')
        gt = self.depth_read(self.gt_list[index])
        depth = self.depth_read(self.depth_list[index])
        return img, depth, gt
        
    def get_length(self):
        return self.__len__()

    def depth_read(self, filename):
        """ Read depth data from file, return as numpy array. """    
        f = open(filename,'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == self.TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(self.TAG_FLOAT,check)
        width = np.fromfile(f,dtype=np.int32,count=1)[0]
        height = np.fromfile(f,dtype=np.int32,count=1)[0]
        size = width*height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
        depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
        return depth

    def depth_write(self, filename, depth):
        """ Write depth to file. """
        height,width = depth.shape[:2]
        f = open(filename,'wb')
        np.array(self.TAG_FLOAT).astype(np.float32).tofile(f)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        depth.astype(np.float32).tofile(f)
        f.close()
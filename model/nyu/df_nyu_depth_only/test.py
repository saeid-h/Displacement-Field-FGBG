import sys, os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from dataloader import get_test_loader
from df import Displacement_Field
from datasets.nyu import NYUDataset

from utils.init_func import init_weight
from misc.utils import get_params
from engine.lr_policy import PolyLR
from engine.engine import Engine

try:
    from imageio import imsave, imread
except:
    from scipy.misc import imsave, imread

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """    
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def depth_write(filename, depth):
    """ Write depth to file. """
    height,width = depth.shape[:2]
    f = open(filename,'wb')
    np.array(TAG_FLOAT).astype(np.float32).tofile(f)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    depth.astype(np.float32).tofile(f)
    f.close()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def save_preds(outpath, image_path, depth, gt):
    CP=[192,192]; RS=[128,160]
    image_name = '_'.join(image_path.split('.')[-2].split('/')[-2:])+'.png'
    
    depth_cmap = depth
    m = np.min(depth_cmap)
    M = np.max(depth_cmap)
    depth_cmap = (depth_cmap - m) / (M - m)
    imsave(os.path.join(outpath, 'cmap' ,image_name), (depth_cmap * 255).astype(np.uint8)) 

    depth_write(os.path.join(outpath, 'raw' ,image_name.replace('.png','.dpt')), depth)

    gt = gt[..., CP[0]:CP[0]+RS[0], CP[1]:CP[1]+RS[1]]
    depth = depth[..., CP[0]:CP[0]+RS[0], CP[1]:CP[1]+RS[1]]
    ref_depth = np.quantile(gt, 0.5)

    occ_gt = np.ones_like(gt)    
    occ_gt[gt > ref_depth] = 0.0
    imsave(os.path.join(outpath, 'occ_mask_gt' ,image_name), (occ_gt * 255).astype(np.uint8))

    occ_init = np.ones_like(gt)    
    occ_init[depth > ref_depth] = 0.0
    imsave(os.path.join(outpath, 'occ_mask_init' ,image_name), (occ_init * 255).astype(np.uint8))

    occ_final = occ_init
    imsave(os.path.join(outpath, 'occ_mask_final' ,image_name), (occ_final * 255).astype(np.uint8))


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    seed = config.seed
    torch.manual_seed(seed)

    test_loader, test_dataset = get_test_loader(engine, NYUDataset, filename_list=args.filename_list)
    config.niters_per_epoch = len(test_loader.dataset)

    BatchNorm2d = nn.BatchNorm2d

    model = Displacement_Field()
    model_dict = model.state_dict()
    trained_model_path = os.path.join(config.root_dir, 'models/nyu_df_bts_d_base', 'epoch-19.pth')
    trained_model_dict = torch.load(trained_model_path, map_location=lambda storage, loc: storage)
    model_weights = {k: v for k, v in trained_model_dict.items() if k in model_dict}
    model_dict.update(model_weights)
    model.load_state_dict(model_dict)
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine.continue_state_object:
        engine.restore_checkpoint()
    model.zero_grad()
    engine.register_state(dataloader=test_loader, model=model)

    if args.save_path:
        outpath = os.path.join(config.root_dir, args.save_path, args.model_name)
        # os.system ('mkdir -p '+os.path.join(outpath, 'rgb'))
        os.system ('mkdir -p '+os.path.join(outpath, 'raw'))
        os.system ('mkdir -p '+os.path.join(outpath, 'cmap'))
        # os.system ('mkdir -p '+os.path.join(outpath, 'depth'))
        os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_init'))
        os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_final'))
        os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_gt'))

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
    dataloader = iter(test_loader)
    for idx in pbar:
        engine.update_iteration(0, idx)
        minibatch = dataloader.next()

        deps = minibatch['data']
        d = deps.numpy()[0,0,...] 
        gt = minibatch['label']
        # masks = minibatch['mask']
        deps = deps.cuda()
        gt = gt.numpy()[0,...]
        pred = model(deps)
        depth = pred.data.cpu().numpy()[0,0,...] 
        
        # print (depth.shape, gt.shape, config.root_dir)
        # print (gt[gt>0], depth[gt>0])
        
        if args.save_path: save_preds(outpath, test_dataset.img_list[idx], depth, gt)

        # fig = plt.figure()
        # ii = plt.imshow(deps[0,0,...].cpu().numpy(), interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()

        print_str = ' Image {}/{}:'.format(idx + 1, config.niters_per_epoch) \

        pbar.set_description(print_str, refresh=False)











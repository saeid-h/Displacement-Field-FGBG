import os.path as osp
import sys, os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F

from config import config
from dataloader import get_train_loader
from df import Displacement_Field
from datasets.nyu import NYUDataset
from datasets.replica import ReplicaDataset

from utils.init_func import init_weight
from misc.utils import get_params
from engine.lr_policy import PolyLR
from engine.engine import Engine

class Mseloss(MSELoss):
    def __init__(self):
        super(Mseloss, self).__init__()

    def forward(self, input, target, mask=None):
        if mask is not None:
            input = input.squeeze(1)
            input = torch.mul(input, mask)
            target = torch.mul(target, mask)
            loss = F.mse_loss(input, target, reduction=self.reduction)
            return loss

class OcclusionLoss(nn.Module):
    def __init__(self, clamp_value=1e-7):
        super(OcclusionLoss, self).__init__()
        self.clamp_value = clamp_value
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, o_pred_logit, o_gt, mask=None):
        o_pred = torch.sigmoid(o_pred_logit)
        if mask is None:
            mask = o_gt.gt(self.clamp_value)
        # o_gt = torch.masked_select(o_gt, mask) 
        # o_pred = torch.masked_select(o_pred, mask)
        o_gt = torch.mul(o_gt, mask)
        o_pred = torch.mul(o_pred, mask) 
        loss = self.loss_fn(o_pred, o_gt) 
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, loss_type='depth', depth_weight=0.5, clamp_value=1e-7):
        super(CombinedLoss, self).__init__()
        self.clamp_value = clamp_value
        self.loss_type = loss_type
        self.depth_weight = depth_weight
        self.depth_loss = Mseloss()
        self.fgbg_los = OcclusionLoss(clamp_value)

    def forward(self, depth, gt, o_pred_logit=None, o_gt=None, mask=None):
        loss = 0
        CP=[96,80]; RS=[128,160]
        fgbg_mask = mask[..., CP[0]:CP[0]+RS[0], CP[1]:CP[1]+RS[1]]
        if self.loss_type == 'depth':
            loss += self.depth_loss(depth, gt, mask)
        if self.loss_type == 'fgbg':
            loss += self.fgbg_los(o_pred_logit, o_gt, fgbg_mask) 
        if self.loss_type == 'combined':
            loss += self.depth_weight * self.depth_loss(depth, gt, mask) + (1. - self.depth_weight) * self.fgbg_los(o_pred_logit, o_gt, fgbg_mask)
        return loss

def get_fgbg(depth, gt, q_min=0.3, q_max=0.7):
    CP=[96,80]; RS=[128,160]
    d_gt_ROI = torch.unsqueeze(gt, 1)[..., CP[0]:CP[0]+RS[0], CP[1]:CP[1]+RS[1]] 
    x_depth_ROI = depth[..., CP[0]:CP[0]+RS[0], CP[1]:CP[1]+RS[1]]

    q30 = np.quantile(d_gt_ROI.cpu().numpy(),q_min, axis=[2,3]).reshape((x_depth_ROI.shape[0],)+(1,)*len(x_depth_ROI.shape[1:]))
    q70 = np.quantile(d_gt_ROI.cpu().numpy(),q_max, axis=[2,3]).reshape((x_depth_ROI.shape[0],)+(1,)*len(x_depth_ROI.shape[1:]))
    ref_depth = torch.as_tensor(np.random.uniform(low=q30,high=q70)).cuda()
    
    gt_offset = ref_depth - d_gt_ROI
    occ_gt = torch.where(gt_offset>0, torch.ones_like(d_gt_ROI), torch.zeros_like(d_gt_ROI))
    fgbg_gt = occ_gt.cuda()
    fgbg_logit = (ref_depth - x_depth_ROI).type(torch.cuda.FloatTensor)
    
    return fgbg_logit, fgbg_gt


parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    seed = config.seed
    torch.manual_seed(seed)

    if args.dataset.lower() == 'nyudv2':
        train_loader, train_sampler = get_train_loader(engine, NYUDataset, args.filename_list)
    elif args.dataset.lower() == 'replica':
        train_loader, train_sampler = get_train_loader(engine, ReplicaDataset, args.filename_list)

    # criterion = Mseloss()
    criterion = CombinedLoss(loss_type=args.loss_type, depth_weight=args.depth_weight)
    BatchNorm2d = nn.BatchNorm2d


    model = Displacement_Field()
    if args.load_ckpt:
        model_dict = model.state_dict()
        trained_model_path = os.path.join(config.root_dir, args.load_ckpt)
        trained_model_dict = torch.load(trained_model_path, map_location=lambda storage, loc: storage)
        model_weights = {k: v for k, v in trained_model_dict.items() if k in model_dict}
        model_dict.update(model_weights)
        model.load_state_dict(model_dict)
    else:
        init_weight(model.displacement_net, nn.init.xavier_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum)

    # model = Displacement_Field()
    # init_weight(model.displacement_net, nn.init.xavier_normal_,
    #             BatchNorm2d, config.bn_eps, config.bn_momentum)
    base_lr = config.lr

    config.niters_per_epoch = len(train_loader.dataset)
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine.continue_state_object:
        engine.restore_checkpoint()
    model.zero_grad()
    model.train()

    optimizer = torch.optim.Adam(params=get_params(model),
                                 lr=base_lr)
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)

    for epoch in range(engine.state.epoch, config.nepochs):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            minibatch = dataloader.next()

            deps = minibatch['data']
            gts = minibatch['label']
            masks = minibatch['mask']

            deps = deps.cuda()
            deps = torch.autograd.Variable(deps)
            gts = gts.cuda()
            gts = torch.autograd.Variable(gts)
            masks = masks.cuda()
            masks = torch.autograd.Variable(masks)
            pred = model(deps)

            if args.loss_type == 'depth':
                loss = criterion(pred, gts, mask=masks)
            else:
                fgbg_logit, fgbg_gt = get_fgbg(pred, gts)
                loss = criterion(pred, gts, fgbg_logit, fgbg_gt, masks)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.6f' % float(loss)

            pbar.set_description(print_str, refresh=False)

        if (epoch == (config.nepochs - 1)) or (epoch % config.snapshot_iter == 0):
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                            config.log_dir,
                                            config.log_dir_link)

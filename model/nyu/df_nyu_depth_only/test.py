import sys
import argparse
from tqdm import tqdm

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

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    seed = config.seed
    torch.manual_seed(seed)

    test_loader, _ = get_test_loader(engine, NYUDataset)
    config.niters_per_epoch = len(test_loader.dataset)

    BatchNorm2d = nn.BatchNorm2d

    model = Displacement_Field()
    init_weight(model.displacement_net, nn.init.xavier_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if engine.continue_state_object:
        engine.restore_checkpoint()
    model.zero_grad()
    engine.register_state(dataloader=test_loader, model=model)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
    dataloader = iter(test_loader)
    for idx in pbar:
        engine.update_iteration(0, idx)
        minibatch = dataloader.next()

        deps = minibatch['data']
        gts = minibatch['label']
        masks = minibatch['mask']

        deps = deps.cuda()
        deps = torch.autograd.Variable(deps)
        gts = gts.cuda()
        gts = torch.autograd.Variable(gts)
        
        pred = model(deps)
        depth = pred.data.cpu().numpy()[0,0,...]
        # fig = plt.figure()
        # ii = plt.imshow(depth, interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()

        print_str = ' Image {}/{}:'.format(idx + 1, config.niters_per_epoch) \

        pbar.set_description(print_str, refresh=False)

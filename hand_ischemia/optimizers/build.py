import os
import sys
from torch import optim
from typing import Set, List, Dict, Any
import logging

__all__ = ['build_optimizer', 'build_lr_scheduler']

logger = logging.getLogger(__name__)


def build_optimizer(cfg, model):
    
    params = list(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(params, lr=cfg.DENOISER.LR, weight_decay=cfg.DENOISER.WEIGHT_DECAY)
    return optimizer


def build_lr_scheduler(cfg, optimizer):

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=0.1)
    return lr_scheduler

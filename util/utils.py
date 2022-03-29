import torch
import numpy as np
import torch.nn as nn
import kornia.morphology as mp
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from util.scheduler_learning_rate import *
import os

def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every mask between two sets of segmentation masks.
    :param set_1: set 1, a tensor of dimensions (1, 224, 224)
    :param set_2: set 2, a tensor of dimensions (1, 224, 224)
    :return: Jaccard Overlap in set 1 with respect to each of the boxes in set 2
    """
    # Find intersections
    intersection = np.logical_and(set_1, set_2)
    # Find the union
    union = np.logical_or(set_1, set_2)
    return intersection / union

def get_img(input, generator):
    with torch.no_grad():
        mask        = generator(input)
        mask_       = mask.view(mask.shape[0], -1)
        mask_min    = mask_.min(1, keepdim=True)[0]
        mask_max    = mask_.max(1, keepdim=True)[0]

        mask        = (mask - mask_min) / (mask_max - mask_min)
        return mask.cpu()

class learning_rate:
    def __init__(self, optimizer, config):
        self.optimizer  = optimizer
        self.config     = config
        self.scheduler  = None

    def get_scheduler(self):
        if self.config.lr_policy == 'lambda':     
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + self.config.epoch_count - self.config.niter) / float(self.config.niter_decay + 1)
                return lr_l
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.config.lr_policy == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config.lr_decay_iters, gamma=0.1)
        elif self.config.lr_policy == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif self.config.lr_policy == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.niter, eta_min=0)
        elif self.config.lr_policy == 'doubleSigmoid':
            self.scheduler = scheduler_learning_rate_sigmoid_double(self.optimizer, self.config.epoch, [0.01, 0.1], [0.1, 0.00001], [10, 10], [0,0])
        elif self.config.lr_policy == 'normal':
            self.scheduler = None
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.config.lr_policy)

    def lr_step(self):
        if self.config.lr_policy == 'doubleSigmoid':
            self.scheduler.step()
        elif self.config.lr_policy == 'normal':
            pass
        else:
            for optimizer in self.optimizer:
                optimizer.step()

from __future__ import print_function
from cgi import test
import os
import math
import secrets
from tkinter.messagebox import NO
from numpy.core.numeric import False_

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from model.generator import *  
from model.simple import *
from model.discriminator import *
from util.loss import *
from util.utils import *
from util.progress_bar import progress_bar
from util.scheduler_learning_rate import *
import numpy as np

from plot.plotChangeObject import plot

class changeObject(object):
    def __init__(self, config, training_loader, val_loader):
        super(changeObject , self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.ganLoss        = GANLoss(use_lsgan=config.lsgan)
        self.regularization = Regularization()
        self.log_interval = config.log
        self.config = config
        self.learning_rate = None
        
        self.train_loader       = training_loader
        self.val_loader         = val_loader

        self.generator = None
        self.classifier = None

        self.plot = None

        self.optimizer = {}

        self.train_loss = []
        self.val_loss = []

    def build_model(self):
        # netG      = 'basic', 'set' : basic for original ResNet, set for InstaGAN's ResNet
        # netD      = 'basic', 'set' : basic for patchGAN discriminator, set for the InstaGAN's discriminator
        # norm      = 'batch', 'instance', 'none'
        # init_type = 'normal', 'xavier', 'kaiming', 'orthogonal'
        self.generator      = define_G(input_nc=1, output_nc=1, ngf=64, netG='basic', norm='batch', init_type='normal', gpu_ids=[self.device])
        # self.discriminator  = define_D(input_nc=1, ndf=64, netD='basic', norm='batch', use_sigmoid=True, init_type='normal', gpu_ids=[self.device])
        self.discriminator  = Discriminator(in_channel=1).to(self.device)

        self.plot = plot(self.train_loader, self.val_loader, self.generator, self.device, self.config)

        self.optimizer['generator']   = torch.optim.SGD(self.generator.parameters(), lr=self.config.lr, weight_decay=0)
        self.optimizer['discriminator'] = torch.optim.SGD(self.discriminator.parameters(), lr=self.config.lr, weight_decay=0)

    def run(self, epoch, data_loader, work):
        if work == 'train':
            self.generator.train()
            self.discriminator.train()
        elif work == 'val':
            self.generator.eval()
            self.discriminator.eval()

        maskSmoothRegular   = 0
        maskRegionRegular   = 0
        discriminatorLoss   = 0
        generatorLoss       = 0
        realAcc             = 0 
        fakeAcc             = 0
        fakeToRealAcc       = 0
        errDrealLoss        = 0
        errDfakeLoss        = 0
        errGfakeLoss        = 0

        iter = 0
        num_data = 0

        for batch_num, (input, ganInput) in enumerate(data_loader):
            # input[0]: input image, input[1]: input image's label
            # ganInput[0]: input image for gan
            iter        += 1
            num_data    += input[0].size(0)
            input       = input[0].to(self.device)
            ganInput    = ganInput[0].to(self.device)

            realLabel       = torch.full((input.size(0),), 1., dtype=torch.float, device=self.device)
            fakeLabel       = torch.full((input.size(0),), 0., dtype=torch.float, device=self.device)
            
            mask = self.generator(input)

            foreground = mask + input
            
            # total variation for smooth, L1 loss for area of region
            # region loss move the mask to keep the output unchanged from the input
            # maskSmooth = self.regularization.tv(mask)
            maskRegion = self.regularization.absoulteRegionLoss(mask)

            # calculate loss with non-thresholded mask
            # regularization = self.config.ms * maskSmooth + self.config.mr * maskRegion
            regularization = self.config.mr * maskRegion

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            realOutput  = self.discriminator(ganInput)
            errD_real   = self.ganLoss(input=realOutput, target_is_real=True)
            fakeOutput  = self.discriminator(foreground.detach())
            errD_fake   = self.ganLoss(input=fakeOutput, target_is_real=False)
            disc_loss   = errD_real + errD_fake

            if work == 'train':
                self.optimizer['discriminator'].zero_grad()
                disc_loss.backward()
                self.optimizer['discriminator'].step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            output      = self.discriminator(foreground)
            errG        = self.ganLoss(input=output, target_is_real=True)
            gener_loss  = errG + regularization

            if work == 'train':
                self.optimizer['generator'].zero_grad()
                gener_loss.backward()
                self.optimizer['generator'].step()

            # maskSmoothRegular   += (maskSmooth.item() * input.size(0))
            maskRegionRegular   += (maskRegion.item() * input.size(0))
            discriminatorLoss   += (disc_loss.item() * input.size(0))
            generatorLoss       += (gener_loss.item() * input.size(0))

            realPred        = (realOutput > 0.5).float()
            realCorrect     = realLabel.eq(realPred.view_as(realLabel)).sum().item()
            realAcc         += realCorrect

            fakePred        = (fakeOutput > 0.5).float()
            fakeCorrect     = fakeLabel.eq(fakePred.view_as(fakeLabel)).sum().item()
            fakeAcc         += fakeCorrect

            fakeToRealPred  = (output > 0.5).float()
            FTRCorrect      = realLabel.eq(fakeToRealPred.view_as(realLabel)).sum().item()
            fakeToRealAcc   += FTRCorrect

            discriminatorLoss   += (disc_loss.item() * input.size(0))
            generatorLoss       += (gener_loss.item() * input.size(0))

            errDfakeLoss        += (errD_fake.item() * input.size(0))
            errDrealLoss        += (errD_real.item() * input.size(0))

            errGfakeLoss        += (errG.item() * input.size(0))

            progress_bar(batch_num, len(data_loader))

        return discriminatorLoss/num_data, generatorLoss/num_data, errDrealLoss/num_data, errDfakeLoss/num_data, errGfakeLoss/num_data, \
                        maskSmoothRegular/num_data, maskRegionRegular/num_data, \
                        100.*realAcc/num_data, 100.*fakeAcc/num_data, 100.*fakeToRealAcc/num_data
                    
        
    def runner(self):

        self.build_model()
        self.learning_rate  = learning_rate(optimizer=self.optimizer, config=self.config)
        self.learning_rate.get_scheduler()

        for i in range(10):
            self.train_loss.append([])
            self.val_loss.append([])
       
        # visualize initialize data
        self.plot.plotResult(epoch=0, trainResult=None, valResult=None)
        
        for epoch in range(1, self.config.epoch + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            
            trainResult = self.run(epoch, self.train_loader, 'train')
            valResult = self.run(epoch, self.val_loader, 'val')

            for i in range(10):
                self.train_loss[i].append(trainResult[i])
                self.val_loss[i].append(valResult[i])
            
            if epoch % self.log_interval == 0 or epoch == 1:
                self.plot.plotResult(epoch,self.train_loss, self.val_loss)

            if epoch == self.config.epoch:
                self.plot.plotResult(epoch, self.train_loss, self.val_loss)

            # scheduler.step()
            self.learning_rate.lr_step()

            for param_group in self.optimizer['generator'].param_groups:
                test = param_group['lr']

            print('=============================={}==============================='.format(test))
            
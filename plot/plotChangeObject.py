import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import sys
from util.utils import *


class plot:
    def __init__(self, train_loader, val_loader, encoder, maskDecoder, device, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.maskDecoder = maskDecoder
        self.device = device
        self.config = config

    def convert_image_np(self, inp, image):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        # mean = np.array((0.485, 0.456, 0.406))
        # std = np.array((0.229, 0.224, 0.225))
        # if image:
        # #     inp = std * inp + mean
        #     inp = np.clip(inp, 0, 1)

        return inp

    def visualize_stn(self, loader):
        with torch.no_grad():
            data,ganData = next(iter(loader))
            data = data[0].to(self.device)
            data = data[:16]
            ganData = ganData[0].to(self.device)
            ganData = ganData[:16]
            input_tensor = data.cpu()
            gan_tensor   = ganData.cpu()

            mask = get_img(data, self.maskDecoder)

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(input_tensor, nrow=4), True)

            gan_grid = self.convert_image_np(
                torchvision.utils.make_grid(gan_tensor, nrow=4), True)
            
            mask_grid = self.convert_image_np(
                torchvision.utils.make_grid(mask, nrow=4), False)

            foreground = mask_grid + in_grid
            
            plt.close('all')

            fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(2, 2)

            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[0,1])

            ax3 = plt.subplot(gs[1,0])
            ax4 = plt.subplot(gs[1,1])

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')

            ax1.imshow(in_grid)
            ax1.set_title('original')
            
            ax2.imshow(gan_grid)
            ax2.set_title('real')

            ax3.imshow(mask_grid)
            ax3.set_title('mask')
            ax4.imshow(foreground)
            ax4.set_title('foreground')

            # # plt.title('{}'.format(count))
            # plt.tight_layout()     
            plt.subplots_adjust(wspace=0.001,hspace=0.2)
            
            return fig

    def visualize_loss(self, trainResult, valResult):
        plt.clf()

        figure, axarr = plt.subplots(2, 5, figsize=(35,7))

        axarr[0][0].plot(trainResult[0], 'r-', label='discriminator loss')
        twin = axarr[0][0].twinx()
        twin.plot(trainResult[1], 'b-', label='generator loss')
        twin.legend(loc='upper right')
        axarr[0][0].legend(loc='upper left')
        axarr[0][0].set_title('train total loss')

        axarr[0][1].plot(trainResult[2], 'r-', label='real to real loss')
        twin = axarr[0][1].twinx()
        twin.plot(trainResult[3], 'b-', label='fake to fake loss')
        twin.legend(loc='upper right')
        axarr[0][1].legend(loc='upper left')
        axarr[0][1].set_title('discriminator')

        axarr[0][2].plot(trainResult[4], 'r-', label='fake to real loss')
        axarr[0][2].legend(loc = 'upper right')
        axarr[0][2].set_title('generator')

        axarr[0][3].plot(trainResult[6], 'r-', label='region reg')
        twin = axarr[0][3].twinx()
        twin.plot(trainResult[5], 'b-', label='tv reg')
        twin.legend(loc='upper right')
        axarr[0][3].set_title('regular')
        axarr[0][3].legend(loc='upper left')

        axarr[0][4].plot(trainResult[7], 'r-', label='real to real acc')
        axarr[0][4].plot(trainResult[8], 'b-', label='fake to fake acc')
        axarr[0][4].plot(trainResult[9], 'g-', label='fake to real acc')
        axarr[0][4].legend(loc='upper left')
        axarr[0][4].set_title('train accuracy')

        axarr[1][0].plot(valResult[0], 'r-', label='discriminator loss')
        twin = axarr[1][0].twinx()
        twin.plot(valResult[1], 'b-', label='generator loss')
        twin.legend(loc='upper right')
        axarr[1][0].legend(loc='upper left')
        axarr[1][0].set_title('val total loss')

        axarr[1][1].plot(valResult[2], 'r-', label='real to real loss')
        twin = axarr[1][1].twinx()
        twin.plot(valResult[3], 'b-', label='fake to fake loss')
        twin.legend(loc='upper right')
        axarr[1][1].legend(loc='upper left')
        axarr[1][1].set_title('discriminator')

        axarr[1][2].plot(valResult[4], 'r-', label='fake to real loss')
        axarr[1][2].legend(loc = 'upper right')
        axarr[1][2].set_title('generator')

        axarr[1][3].plot(valResult[6], 'r-', label='region reg')
        twin = axarr[1][3].twinx()
        twin.plot(valResult[5], 'b-', label='tv reg')
        twin.legend(loc='upper right')
        axarr[1][3].set_title('regular')
        axarr[1][3].legend(loc='upper left')

        axarr[1][4].plot(valResult[7], 'r-', label='real to real acc')
        axarr[1][4].plot(valResult[8], 'b-', label='fake to fake acc')
        axarr[1][4].plot(valResult[9], 'g-', label='fake to real acc')
        axarr[1][4].legend(loc='upper left')
        axarr[1][4].set_title('val accuracy')

        plt.tight_layout()    

        return figure

    def plotResult(self, epoch, trainResult, valResult):
        path = os.path.join('./result/circleRectangle/checkIdeaWorking', self.config.title + \
                   '_mr_' + str(self.config.mr) + '_ms_' + str(self.config.ms))

        # path = os.path.join('../../../result/test/')

        if epoch != 0: 
            # visualize train data
            trainPicPath = os.path.join(path, 'pic/train')
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/val')
            trainPic2 = self.visualize_stn(self.val_loader)
            try:
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')
            except FileNotFoundError:
                os.makedirs(valPicPath)
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')

            # visualize loss graph
            lossPath = os.path.join(path, 'graph')
            loss = self.visualize_loss(trainResult, valResult)
            try:
                loss.savefig(os.path.join(lossPath + '/loss.png'),bbox_inches='tight')
            except FileNotFoundError:
                os.makedirs(lossPath)
                loss.savefig(os.path.join(lossPath + '/loss.png'),bbox_inches='tight')

        elif epoch == 0:
             # visualize train data
            trainPicPath = os.path.join(path, 'pic/train')
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/val')
            trainPic2 = self.visualize_stn(self.val_loader)
            try:
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')
            except FileNotFoundError:
                os.makedirs(valPicPath)
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)),bbox_inches='tight')

            
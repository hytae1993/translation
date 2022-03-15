from numpy import ma
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
            data,_ = next(iter(loader))
            data = data.to(self.device)
            data = data[:16]
            input_tensor = data.cpu()

            mask, threshold_mask, erode, dilate, band, foreSmooth = get_mum_image_band_multi(data, self.encoder, self.maskDecoder, self.config.iter, self.config.last)


            fig, axarr = plt.subplots(self.config.last+1, 5, figsize=((self.config.last+1)*5, 5*5))

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(input_tensor, nrow=4), True)
            axarr[0, 2].imshow(in_grid)
            axarr[0, 2].axis('off')
            axarr[0, 0].axis('off')
            axarr[0, 1].axis('off')
            axarr[0, 3].axis('off')
            axarr[0, 4].axis('off')
            
            for i in range(1, self.config.last+1):
                axarr[i, 0].imshow(self.convert_image_np(
                    torchvision.utils.make_grid(mask[i-1], nrow=4), True))
                axarr[i, 0].set_title('mask {}'.format(i))
                axarr[i, 0].axis('off')

                axarr[i, 1].imshow(self.convert_image_np(
                    torchvision.utils.make_grid(threshold_mask[i-1], nrow=4), True))
                axarr[i, 1].set_title('threshold mask {}'.format(i))
                axarr[i, 1].axis('off')

                axarr[i, 2].imshow(self.convert_image_np(
                    torchvision.utils.make_grid(band[i-1], nrow=4), True))
                axarr[i, 2].set_title('band {}'.format(i))
                axarr[i, 2].axis('off')

                axarr[i, 3].imshow(self.convert_image_np(
                    torchvision.utils.make_grid(foreSmooth[i-1], nrow=4), True))
                axarr[i, 3].set_title('smooth {}_iter_{}'.format(i, self.config.iter))
                axarr[i, 3].axis('off')

                axarr[i, 4].imshow(self.convert_image_np(
                    torchvision.utils.make_grid(mask[i-1]*foreSmooth[i-1], nrow=4), True))
                axarr[i, 4].set_title('mask * smooth {}'.format(i))
                axarr[i, 4].axis('off')


            return fig

    def visualize_loss(self, trainResult, valResult):
        color = ['r-','g-','b-','c-','k-','m-','y-', 'indigo-']
        plt.clf()
        figure, axarr = plt.subplots(1, 6, figsize=(48,8))

        axarr[0].plot(trainResult[0], 'r-', label='train loss')
        axarr[0].fill_between(range(len(trainResult[0])), np.array(trainResult[0])-np.array(trainResult[1]), np.array(trainResult[0])+np.array(trainResult[1]),alpha=.1, color='r')
        axarr[0].plot(valResult[0], 'b-', label='val loss')
        axarr[0].fill_between(range(len(valResult[0])), np.array(valResult[0])-np.array(valResult[1]), np.array(valResult[0])+np.array(valResult[1]),alpha=.1, color='b')
        axarr[0].legend(loc='upper left')
        axarr[0].set_title('total loss')

        axarr[1].plot(trainResult[2], 'r-', label='train reg')
        axarr[1].plot(valResult[2], 'g-', label='val reg')
        twin = axarr[1].twinx()
        twin.plot(trainResult[3], 'b-', label='train seg')
        twin.plot(valResult[3], 'c-', label='val seg')
        twin.legend(loc='upper left')
        axarr[1].legend(loc='upper right')
        axarr[1].set_title('seg loss and reg')

        trainMumfordLoss = list(map(list, zip(*trainResult[4])))
        valMumfordLoss = list(map(list, zip(*valResult[4])))
        trainReg = list(map(list, zip(*trainResult[5])))
        valReg = list(map(list, zip(*valResult[5])))

        for i in range(self.config.last):
            axarr[2].plot(trainMumfordLoss[i], color[i], label='mask {}'.format(i+1))
            axarr[2].set_title('train mumford loss')
            axarr[2].legend(loc='upper left')

        for i in range(self.config.last):
            axarr[3].plot(valMumfordLoss[i], color[i], label='mask {}'.format(i+1))
            axarr[3].set_title('val mumford loss')
            axarr[3].legend(loc='upper left')

        for i in range(self.config.last):
            axarr[4].plot(trainReg[i], color[i], label='mask {}'.format(i+1))
            axarr[4].set_title('train reg loss')
            axarr[4].legend(loc='upper left')

        for i in range(self.config.last):
            axarr[5].plot(valReg[i], color[i], label='mask {}'.format(i+1))
            axarr[5].set_title('val reg loss')
            axarr[5].legend(loc='upper left')
        

        plt.tight_layout()    

        return figure

    def plotResult(self, epoch, trainResult, valResult):
        path = os.path.join('../../../result/mask/dogcat/band/test2/multi_mask', self.config.title + 'phi_' + str(self.config.last) + '/_iter_' + str(self.config.iter) + '/_ms_' + str(self.config.ms))

        if epoch != 0: 
            # visualize train data
            trainPicPath = os.path.join(path, 'pic/train')
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/val')
            trainPic2 = self.visualize_stn(self.val_loader)
            try:
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(valPicPath)
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))

            # visualize loss graph
            lossPath = os.path.join(path, 'graph')
            loss = self.visualize_loss(trainResult, valResult)
            try:
                loss.savefig(os.path.join(lossPath + '/loss.png'))
            except FileNotFoundError:
                os.makedirs(lossPath)
                loss.savefig(os.path.join(lossPath + '/loss.png'))

        elif epoch == 0:
             # visualize train data
            trainPicPath = os.path.join(path, 'pic/train')
            trainPic1 = self.visualize_stn(self.train_loader)
            try:
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(trainPicPath)
                trainPic1.savefig(os.path.join(trainPicPath + '/result_{}.png'.format(epoch)))

            # visualize validation data
            valPicPath = os.path.join(path, 'pic/val')
            trainPic2 = self.visualize_stn(self.val_loader)
            try:
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))
            except FileNotFoundError:
                os.makedirs(valPicPath)
                trainPic2.savefig(os.path.join(valPicPath + '/result_{}.png'.format(epoch)))

        plt.clf()
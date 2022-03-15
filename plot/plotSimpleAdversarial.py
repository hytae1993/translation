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

            mask = get_mum_image2(data, self.encoder, self.maskDecoder)
            threshold_mask = get_threshold_mask(mask)

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(input_tensor, nrow=4), True)

            gan_grid = self.convert_image_np(
                torchvision.utils.make_grid(gan_tensor, nrow=4), True)
            
            mask_grid = self.convert_image_np(
                torchvision.utils.make_grid(mask, nrow=4), False)
            
            fore_grid = in_grid * (mask_grid)
            back_grid = in_grid * (1-mask_grid)

            threshold_mask_grid = self.convert_image_np(
                torchvision.utils.make_grid(threshold_mask, nrow=4), False)


            plt.close('all')
            fig = plt.figure()
            # fig.tight_layout()
            ax1 = fig.add_subplot(2,3,1)
            ax2 = fig.add_subplot(2,3,2)
            ax3 = fig.add_subplot(2,3,3)
            ax4 = fig.add_subplot(2,3,4)
            ax5 = fig.add_subplot(2,3,5)
            ax6 = fig.add_subplot(2,3,6)
            

            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)
            ax4.axes.get_xaxis().set_visible(False)
            ax4.axes.get_yaxis().set_visible(False)
            ax5.axes.get_xaxis().set_visible(False)
            ax5.axes.get_yaxis().set_visible(False)
            ax6.axes.get_xaxis().set_visible(False)
            ax6.axes.get_yaxis().set_visible(False)

            ax1.imshow(in_grid)
            ax1.set_title('original')
            ax2.imshow(mask_grid, cmap='gray', vmin=0, vmax=1)
            ax2.set_title('mask')
            ax3.imshow(threshold_mask_grid, cmap='gray', vmin=0, vmax=1)
            ax3.set_title('threshold mask')
            ax4.imshow(fore_grid)
            ax4.set_title('mask * input')
            ax5.imshow(threshold_mask_grid * in_grid)
            ax5.set_title('thresholded mask * input')
            ax6.imshow(gan_grid)
            ax6.set_title('real input')

            # plt.title('{}'.format(count))
            plt.tight_layout()     
            plt.subplots_adjust(wspace=0.01,hspace=0.3)

            return fig

    def visualize_loss(self, trainResult, valResult):
        plt.clf()
        figure, axarr = plt.subplots(2, 4, figsize=(24,8))

        axarr[0][0].plot(trainResult[0], 'r-', label='discriminator loss')
        twin = axarr[0][0].twinx()
        twin.plot(trainResult[1], 'b-', label='generator loss')
        twin.legend(loc='upper right')
        axarr[0][0].legend(loc='upper left')
        axarr[0][0].set_title('train total loss')

        axarr[0][1].plot(trainResult[6], 'r-', label='real to real loss')
        twin = axarr[0][1].twinx()
        twin.plot(trainResult[7], 'b-', label='fake to fake loss')
        twin.legend(loc='upper right')
        axarr[0][1].legend(loc='upper left')
        axarr[0][1].set_title('discriminator')

        axarr[0][2].plot(trainResult[2], 'r-', label='region reg')
        axarr[0][2].set_title('regular')
        axarr[0][2].legend(loc='upper left')

        axarr[0][3].plot(trainResult[3], 'r-', label='real to real acc')
        axarr[0][3].plot(trainResult[4], 'b-', label='fake to fake acc')
        axarr[0][3].plot(trainResult[5], 'g-', label='fake to real acc')
        axarr[0][3].legend(loc='upper left')
        axarr[0][3].set_title('train accuracy')

        axarr[1][0].plot(valResult[0], 'r-', label='discriminator loss')
        twin = axarr[1][0].twinx()
        twin.plot(valResult[1], 'b-', label='generator loss')
        twin.legend(loc='upper right')
        axarr[1][0].legend(loc='upper left')
        axarr[1][0].set_title('val total loss')

        axarr[1][1].plot(valResult[6], 'r-', label='real to real loss')
        twin = axarr[1][1].twinx()
        twin.plot(valResult[7], 'b-', label='fake to fake loss')
        twin.legend(loc='upper right')
        axarr[1][1].legend(loc='upper left')
        axarr[1][1].set_title('discriminator')

        axarr[1][2].plot(valResult[2], 'r-', label='region reg')
        axarr[1][2].set_title('regular')
        axarr[1][2].legend(loc='upper left')

        axarr[1][3].plot(valResult[3], 'r-', label='real to real acc')
        axarr[1][3].plot(valResult[4], 'b-', label='fake to fake acc')
        axarr[1][3].plot(valResult[5], 'g-', label='fake to real acc')
        axarr[1][3].legend(loc='upper left')
        axarr[1][3].set_title('val accuracy')


        plt.tight_layout()    

        return figure

    def plotResult(self, epoch, trainResult, valResult):
        path = os.path.join('../../../result/doctor/bi-partitioning/circle/GANwithoutMumford/withDisc/cat/mrChange', self.config.title + '_mr_' + str(self.config.mr))
        # path = os.path.join('../../../result/', self.config.title + '_mr_' + str(self.config.mr))

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

            
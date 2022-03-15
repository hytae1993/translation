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

            mask, dilate, erode, band, foreground, background, foreSmooth, backSmooth = get_mum_image_band(data, self.encoder, self.maskDecoder, self.config.iter)
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

            threshold_fore_grid = in_grid * threshold_mask_grid
            threshold_back_grid = in_grid * (1 - threshold_mask_grid)

            dilate_grid = self.convert_image_np(
                torchvision.utils.make_grid(dilate, nrow=4), False)
            erode_grid = self.convert_image_np(
                torchvision.utils.make_grid(erode, nrow=4), False)
            band_grid = self.convert_image_np(
                torchvision.utils.make_grid(band, nrow=4), False)

            fore_matrix_grid = self.convert_image_np(
                torchvision.utils.make_grid(foreground, nrow=4), False)
            back_matrix_grid = self.convert_image_np(
                torchvision.utils.make_grid(background, nrow=4), False)
            fore_smooth_grid = self.convert_image_np(
                torchvision.utils.make_grid(foreSmooth, nrow=4), False)
            back_smooth_grid = self.convert_image_np(
                torchvision.utils.make_grid(backSmooth, nrow=4), False)
            
            plt.close('all')


            fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(5, 8)

            ax1 = plt.subplot(gs[0,2:4])
            ax2 = plt.subplot(gs[0,4:6])

            ax3 = plt.subplot(gs[1,1:3])
            ax4 = plt.subplot(gs[1,3:5])
            ax5 = plt.subplot(gs[1,5:7])

            ax6 = plt.subplot(gs[2,1:3])
            ax7 = plt.subplot(gs[2,3:5])
            ax8 = plt.subplot(gs[2,5:7])

            ax9 = plt.subplot(gs[3,1:3])
            ax10 = plt.subplot(gs[3,3:5])
            ax11 = plt.subplot(gs[3,5:7])

            ax12 = plt.subplot(gs[4,0:2])
            ax13 = plt.subplot(gs[4,2:4])
            ax14 = plt.subplot(gs[4,4:6])
            ax15 = plt.subplot(gs[4,6:])

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            ax4.axis('off')
            ax5.axis('off')
            ax6.axis('off')
            ax7.axis('off')
            ax8.axis('off')
            ax9.axis('off')
            ax10.axis('off')
            ax11.axis('off')
            ax12.axis('off')
            ax13.axis('off')
            ax14.axis('off')
            ax15.axis('off')

            plt.subplots_adjust(wspace=0.3, hspace=0.1)
            # fig.tight_layout(pad=1.0)

            ax1.imshow(in_grid)
            ax1.set_title('original')
            
            ax2.imshow(gan_grid)
            ax2.set_title('real')

            maskPic = ax3.imshow(mask_grid, cmap='gray', vmin=0, vmax=1)
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(maskPic, cax=cax, orientation='vertical')
            ax3.set_title('mask')
            ax4.imshow(fore_grid)
            ax4.set_title('foreground')
            ax5.imshow(back_grid)
            ax5.set_title('background')

            ax6.imshow(threshold_mask_grid, cmap='gray', vmin=0, vmax=1)
            ax6.set_title('thresholded mask')
            ax7.imshow(threshold_fore_grid)
            ax7.set_title('thresholded foreground')
            ax8.imshow(threshold_back_grid)
            ax8.set_title('thresholded background')

            ax9.imshow(dilate_grid, cmap='gray', vmin=0, vmax=1)
            ax9.set_title('dilate')
            ax10.imshow(erode_grid, cmap='gray', vmin=0, vmax=1)
            ax10.set_title('erode')
            ax11.imshow(band_grid, cmap='gray', vmin=0, vmax=1)
            ax11.set_title('band')

            ax12.imshow(fore_matrix_grid)
            ax12.set_title('dilate * input')
            ax13.imshow(back_matrix_grid)
            ax13.set_title('erode * input')
            ax14.imshow(fore_smooth_grid)
            ax14.set_title('foreground smooth')
            ax15.imshow(back_smooth_grid)
            ax15.set_title('background smooth')

            # # plt.title('{}'.format(count))
            # plt.tight_layout()     
            plt.subplots_adjust(wspace=0.001,hspace=0.2)
            
            return fig

    def visualize_loss(self, trainResult, valResult):
        plt.clf()

        figure, axarr = plt.subplots(2, 6, figsize=(42,7))

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

        axarr[0][3].plot(trainResult[9], 'r-', label='region reg')
        twin = axarr[0][3].twinx()
        twin.plot(trainResult[8], 'b-', label='tv reg')
        twin.legend(loc='upper right')
        axarr[0][3].set_title('regular')
        axarr[0][3].legend(loc='upper left')

        axarr[0][4].plot(trainResult[7], 'r-', label='mumford')
        twin = axarr[0][4].twinx()
        twin.plot(trainResult[5], 'b-', label='foreground')
        twin.plot(trainResult[6], 'g-', label='background')
        twin.legend(loc='upper right')
        axarr[0][4].legend(loc='upper left')
        axarr[0][4].set_title('segmentation')

        axarr[0][5].plot(trainResult[10], 'r-', label='real to real acc')
        axarr[0][5].plot(trainResult[11], 'b-', label='fake to fake acc')
        axarr[0][5].plot(trainResult[12], 'g-', label='fake to real acc')
        axarr[0][5].legend(loc='upper left')
        axarr[0][5].set_title('train accuracy')

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

        axarr[1][3].plot(valResult[9], 'r-', label='region reg')
        twin = axarr[1][3].twinx()
        twin.plot(valResult[8], 'b-', label='tv reg')
        twin.legend(loc='upper right')
        axarr[1][3].set_title('regular')
        axarr[1][3].legend(loc='upper left')

        axarr[1][4].plot(valResult[7], 'r-', label='mumford')
        twin = axarr[1][4].twinx()
        twin.plot(valResult[5], 'b-', label='foreground')
        twin.plot(valResult[6], 'g-', label='background')
        twin.legend(loc='upper right')
        axarr[1][4].legend(loc='upper left')
        axarr[1][4].set_title('segmentation')

        axarr[1][5].plot(valResult[10], 'r-', label='real to real acc')
        axarr[1][5].plot(valResult[11], 'b-', label='fake to fake acc')
        axarr[1][5].plot(valResult[12], 'g-', label='fake to real acc')
        axarr[1][5].legend(loc='upper left')
        axarr[1][5].set_title('val accuracy')


        plt.tight_layout()    

        return figure

    def plotResult(self, epoch, trainResult, valResult):
        path = os.path.join('../../../result/doctor/bi-partitioning/bird/test/', self.config.title + '_iter_' + str(self.config.iter) + \
                   '/_mt_' + str(self.config.mt) + '_ms_' + str(self.config.ms))

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

            
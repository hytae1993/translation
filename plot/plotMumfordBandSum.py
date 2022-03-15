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

            mask, dilate, erode, band, foreground, background, foreSmoothList, backSmoothList = get_mum_image_band_sum(data, self.encoder, self.maskDecoder)
            threshold_mask = get_threshold_mask(mask)

            in_grid = self.convert_image_np(
                torchvision.utils.make_grid(input_tensor, nrow=4), True)
            
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


            fore_smooth_grid1 = self.convert_image_np(
                torchvision.utils.make_grid(foreSmoothList[0].cpu(), nrow=4), False)
            back_smooth_grid1 = self.convert_image_np(
                torchvision.utils.make_grid(backSmoothList[0].cpu(), nrow=4), False)
            th_complement_grid1 = fore_smooth_grid1 * threshold_mask_grid + back_smooth_grid1 * (1-threshold_mask_grid)
            ma_complement_grid1 = fore_smooth_grid1 * mask_grid + back_smooth_grid1 * (1-mask_grid)

            fore_smooth_grid2 = self.convert_image_np(
                torchvision.utils.make_grid(foreSmoothList[1].cpu(), nrow=4), False)
            back_smooth_grid2 = self.convert_image_np(
                torchvision.utils.make_grid(backSmoothList[1].cpu(), nrow=4), False)
            th_complement_grid2 = fore_smooth_grid2 * threshold_mask_grid + back_smooth_grid2 * (1-threshold_mask_grid)
            ma_complement_grid2 = fore_smooth_grid2 * mask_grid + back_smooth_grid2 * (1-mask_grid)

            fore_smooth_grid3 = self.convert_image_np(
                torchvision.utils.make_grid(foreSmoothList[2].cpu(), nrow=4), False)
            back_smooth_grid3 = self.convert_image_np(
                torchvision.utils.make_grid(backSmoothList[2].cpu(), nrow=4), False)
            th_complement_grid3 = fore_smooth_grid3 * threshold_mask_grid + back_smooth_grid3 * (1-threshold_mask_grid)
            ma_complement_grid3 = fore_smooth_grid3 * mask_grid + back_smooth_grid3 * (1-mask_grid)

            fore_smooth_grid4 = self.convert_image_np(
                torchvision.utils.make_grid(foreSmoothList[3].cpu(), nrow=4), False)
            back_smooth_grid4 = self.convert_image_np(
                torchvision.utils.make_grid(backSmoothList[3].cpu(), nrow=4), False)
            th_complement_grid4 = fore_smooth_grid4 * threshold_mask_grid + back_smooth_grid4 * (1-threshold_mask_grid)
            ma_complement_grid4 = fore_smooth_grid4 * mask_grid + back_smooth_grid4 * (1-mask_grid)

            plt.close('all')
            fig = plt.figure(figsize=(40,20))
            fig.tight_layout()
            ax1 = fig.add_subplot(8,3,2)

            ax5 = fig.add_subplot(8,3,4)
            ax6 = fig.add_subplot(8,3,5)
            ax7 = fig.add_subplot(8,3,6)

            ax8 = fig.add_subplot(8,3,7)
            ax9 = fig.add_subplot(8,3,8)
            ax10 = fig.add_subplot(8,3,9)

            ax11 = fig.add_subplot(8,3,10)
            ax12 = fig.add_subplot(8,3,11)
            ax13 = fig.add_subplot(8,3,12)
            
            ax14 = fig.add_subplot(8,4,17)
            ax15 = fig.add_subplot(8,4,18)
            ax16 = fig.add_subplot(8,4,19)
            ax17 = fig.add_subplot(8,4,20)

            ax18 = fig.add_subplot(8,4,21)
            ax19 = fig.add_subplot(8,4,22)
            ax20 = fig.add_subplot(8,4,23)
            ax21 = fig.add_subplot(8,4,24)

            ax22 = fig.add_subplot(8,4,25)
            ax23 = fig.add_subplot(8,4,26)
            ax24 = fig.add_subplot(8,4,27)
            ax25 = fig.add_subplot(8,4,28)

            ax26 = fig.add_subplot(8,4,29)
            ax27 = fig.add_subplot(8,4,30)
            ax28 = fig.add_subplot(8,4,31)
            ax29 = fig.add_subplot(8,4,32)

            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            ax5.axes.get_xaxis().set_visible(False)
            ax5.axes.get_yaxis().set_visible(False)
            ax6.axes.get_xaxis().set_visible(False)
            ax6.axes.get_yaxis().set_visible(False)
            ax7.axes.get_xaxis().set_visible(False)
            ax7.axes.get_yaxis().set_visible(False)
            ax8.axes.get_xaxis().set_visible(False)
            ax8.axes.get_yaxis().set_visible(False)
            ax9.axes.get_xaxis().set_visible(False)
            ax9.axes.get_yaxis().set_visible(False)
            ax10.axes.get_xaxis().set_visible(False)
            ax10.axes.get_yaxis().set_visible(False)
            ax11.axes.get_xaxis().set_visible(False)
            ax11.axes.get_yaxis().set_visible(False)
            ax12.axes.get_xaxis().set_visible(False)
            ax12.axes.get_yaxis().set_visible(False)
            ax13.axes.get_xaxis().set_visible(False)
            ax13.axes.get_yaxis().set_visible(False)
            ax14.axes.get_xaxis().set_visible(False)
            ax14.axes.get_yaxis().set_visible(False)
            ax15.axes.get_xaxis().set_visible(False)
            ax15.axes.get_yaxis().set_visible(False)
            ax16.axes.get_xaxis().set_visible(False)
            ax16.axes.get_yaxis().set_visible(False)
            ax17.axes.get_xaxis().set_visible(False)
            ax17.axes.get_yaxis().set_visible(False)
            ax18.axes.get_xaxis().set_visible(False)
            ax18.axes.get_yaxis().set_visible(False)
            ax19.axes.get_xaxis().set_visible(False)
            ax19.axes.get_yaxis().set_visible(False)
            ax20.axes.get_xaxis().set_visible(False)
            ax20.axes.get_yaxis().set_visible(False)
            ax21.axes.get_xaxis().set_visible(False)
            ax21.axes.get_yaxis().set_visible(False)
            ax22.axes.get_xaxis().set_visible(False)
            ax22.axes.get_yaxis().set_visible(False)
            ax23.axes.get_xaxis().set_visible(False)
            ax23.axes.get_yaxis().set_visible(False)
            ax24.axes.get_xaxis().set_visible(False)
            ax24.axes.get_yaxis().set_visible(False)
            ax25.axes.get_xaxis().set_visible(False)
            ax25.axes.get_yaxis().set_visible(False)
            ax26.axes.get_xaxis().set_visible(False)
            ax26.axes.get_yaxis().set_visible(False)
            ax27.axes.get_xaxis().set_visible(False)
            ax27.axes.get_yaxis().set_visible(False)
            ax28.axes.get_xaxis().set_visible(False)
            ax28.axes.get_yaxis().set_visible(False)
            ax29.axes.get_xaxis().set_visible(False)
            ax29.axes.get_yaxis().set_visible(False)

            ax1.imshow(in_grid)
            ax1.set_title('original')

            ax5.imshow(threshold_mask_grid, cmap='gray', vmin=0, vmax=1)
            ax5.set_title('thresholded mask')
            ax6.imshow(threshold_fore_grid)
            ax6.set_title('thresholded foreground')
            ax7.imshow(threshold_back_grid)
            ax7.set_title('thresholded background')

            maskPic = ax8.imshow(mask_grid, cmap='gray', vmin=0, vmax=1)
            divider = make_axes_locatable(ax8)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(maskPic, cax=cax, orientation='vertical')
            ax8.set_title('mask')
            ax9.imshow(fore_grid)
            ax9.set_title('foreground')
            ax10.imshow(back_grid)
            ax10.set_title('background')

            ax11.imshow(dilate_grid, cmap='gray', vmin=0, vmax=1)
            ax11.set_title('dilate')
            ax12.imshow(erode_grid, cmap='gray', vmin=0, vmax=1)
            ax12.set_title('erode')
            ax13.imshow(band_grid, cmap='gray', vmin=0, vmax=1)
            ax13.set_title('band')

            ax14.imshow(fore_smooth_grid1)
            ax14.set_title('foreground smooth 8')
            ax15.imshow(back_smooth_grid1)
            ax15.set_title('background smooth 8')
            ax16.imshow(th_complement_grid1)
            ax16.set_title('threshold complement 8')
            ax17.imshow(ma_complement_grid1)
            ax17.set_title('mask complement 8')

            ax18.imshow(fore_smooth_grid2)
            ax18.set_title('foreground smooth 16')
            ax19.imshow(back_smooth_grid2)
            ax19.set_title('background smooth 16')
            ax20.imshow(th_complement_grid2)
            ax20.set_title('threshold complement 16')
            ax21.imshow(ma_complement_grid2)
            ax21.set_title('mask complement 16')

            ax22.imshow(fore_smooth_grid3)
            ax22.set_title('foreground smooth 24')
            ax23.imshow(back_smooth_grid3)
            ax23.set_title('background smooth 24')
            ax24.imshow(th_complement_grid3)
            ax24.set_title('threshold complement 24')
            ax25.imshow(ma_complement_grid3)
            ax25.set_title('mask complement 24')

            ax26.imshow(fore_smooth_grid4)
            ax26.set_title('foreground smooth 32')
            ax27.imshow(back_smooth_grid4)
            ax27.set_title('background smooth 32')
            ax28.imshow(th_complement_grid4)
            ax28.set_title('threshold complement 32')
            ax29.imshow(ma_complement_grid4)
            ax29.set_title('mask complement 32')

            # plt.title('{}'.format(count))
            plt.tight_layout()     

            return fig

    def visualize_loss(self, trainResult, valResult):
        plt.clf()
        figure, axarr = plt.subplots(1, 3, figsize=(24,8))

        axarr[0].plot(trainResult[0], 'r-', label='train loss')
        axarr[0].fill_between(range(len(trainResult[0])), np.array(trainResult[0])-np.array(trainResult[1]), np.array(trainResult[0])+np.array(trainResult[1]),alpha=.1, color='r')
        axarr[0].plot(valResult[0], 'b-', label='val loss')
        axarr[0].fill_between(range(len(valResult[0])), np.array(valResult[0])-np.array(valResult[1]), np.array(valResult[0])+np.array(valResult[1]),alpha=.1, color='b')
        axarr[0].legend(loc='upper left')
        axarr[0].set_title('total loss')

        axarr[1].plot(trainResult[3], 'r-', label='train foreground mumford')
        axarr[1].plot(valResult[3], 'g-', label='val foreground mumford')
        axarr[1].plot(trainResult[4], 'b-', label='train background mumford')
        axarr[1].plot(valResult[4], 'c-', label='val background mumford')
        axarr[1].set_title('mumford loss')
        axarr[1].legend(loc='upper right')
        
        axarr[2].plot(trainResult[2], 'g-', label='train mask smooth reg')
        axarr[2].plot(valResult[2], 'c-', label='val mask smooth reg')
        axarr[2].set_title('regular')
        axarr[2].legend(loc='upper left')

        plt.tight_layout()    

        return figure

    def plotResult(self, epoch, trainResult, valResult):
        path = os.path.join('../../../result/mask/dogcat/band/test2/laplaceSum', self.config.title + '_ms_' + str(self.config.ms))

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

            
from __future__ import print_function

import argparse
import os
import sys
from numpy.core.defchararray import split

import torch
# from torch.utils.data.dataset import ConcatDataset
from util.dataUtil import *
from torchvision import datasets, transforms
import numpy as np
import random

from util.utils import *
from util.dataUtil import *

from changeObject import changeObject

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch segmentation')
# hyper-parameters
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log', type=int, default=1)
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--ms', type=float, default=0.01, help='mask smooth')
parser.add_argument('--mr', type=float, default=0.01, help='mask region')
parser.add_argument('--gp', type=float, default=1, help='gradient penalty')
parser.add_argument('--lsgan', type=bool, default=False, help='using lsgan')
parser.add_argument('--lr_policy', type=str, default='normal', help='learning rate scheduler')

# model configuration
parser.add_argument('--model', '-m', type=str, default='changeObject', help='choose which model is going to use')
parser.add_argument('--title', default='', type=str, help='title for the saved component')

args = parser.parse_args()

class main:
    def __init__(self):
        self.model              = None
        self.train_loader       = None
        self.val_loader         = None
        self.test_loader        = None
        self.train_gan_loader   = None
        self.val_gan_loader     = None


    def dataLoad(self):
        # ===========================================================
        # Set train dataset & validation dataset
        # ===========================================================
        print('===> Loading datasets')

        def seed_worker(self):
            np.random.seed(args.seed)
            random.seed(args.seed)

        if args.model == 'backgroundGAN' or args.model == 'simpleAdversarial' or args.model == 'changeObject':
            self.train_loader = torch.utils.data.DataLoader(
                CustomConcatDataset(
                    datasets.ImageFolder(
                        root='../../../dataset/dog/train',
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Grayscale(),
                        ])), 
                    datasets.ImageFolder(
                        root='../../../dataset/cat/train',
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Grayscale(),
                        ]))
                ),
                batch_size=args.batchSize, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=seed_worker, )
            
            self.val_loader = torch.utils.data.DataLoader(
                CustomConcatDataset(
                    datasets.ImageFolder(
                            root='../../../dataset/dog/val',
                            transform=transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ])),
                    datasets.ImageFolder(
                            root='../../../dataset/cat/val', 
                            transform=transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Grayscale(),
                            ]))
                ),
                batch_size=args.batchSize, shuffle=False, num_workers=2, drop_last=True, worker_init_fn=seed_worker)

        else:
            self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                        root='../../../dataset/cat/train',
                                # root='../../dataset/GTSRB/Training',
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                        ])), 
                        batch_size=args.batchSize, shuffle=True, num_workers=2, drop_last=True, worker_init_fn=seed_worker)
            self.val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                            root='../../../dataset/cat/val', 
                            transform=transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])), 
                            batch_size=args.batchSize, shuffle=False, num_workers=2, drop_last=True, worker_init_fn=seed_worker)
                 
    def modelCall(self):
        
        if args.model == 'changeObject': 
            self.model = changeObject(args, self.train_loader, self.val_loader)
        else:
            sys.exit("Need to speicify the model")

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

    main = main()
    main.dataLoad()
    main.modelCall()
    
    main.model.runner()
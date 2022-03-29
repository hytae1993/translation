from pickle import FALSE
import torch
from torch import nn
from torchvision.datasets import * # Training dataset
from torchvision import models, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.parameter import Parameter

import argparse
import configparser
import sys
import os
import datetime 
from tqdm import tqdm
import copy

dir_utility = '/nas/users/hong/work/utility/'
sys.path.append(dir_utility)

from network.network import Network_Discriminator
from network.network import Network_Generator
from visualize.visualize import *
from intensity.transform import * 
from optimize.scheduler_learning_rate import *
from evaluate.logger import Logger

# ======================================================================
# usage:
#   python3 main.py OPTION_FILE CUDA_DEVICE_NUMBER
#   python3 main.py option_file.ini 0
# ======================================================================
if len(sys.argv) == 1:
    file_option         = './option/option_default.ini'
    cuda_device_number  = 0

elif len(sys.argv) == 2:
    file_option         = sys.argv[1]
    cuda_device_number  = 0

elif len(sys.argv) == 3:
    file_option         = sys.argv[1]
    cuda_device_number  = int(sys.argv[2])

else:
    pass

# ======================================================================
# options
# ======================================================================
config = configparser.ConfigParser()
config.read(file_option)

option_dataset_real = {
    'name'              : config.get('Dataset.real', 'Dataset.real.name'),
    'directory'         : config.get('Dataset.real', 'Dataset.real.directory'),
    'channel'           : int(config.get('Dataset.real', 'Dataset.real.channel')),
    'height'            : int(config.get('Dataset.real', 'Dataset.real.height')),
    'width'             : int(config.get('Dataset.real', 'Dataset.real.width')),
    'normalize'         : config.get('Dataset.real', 'Dataset.real.normalize'), 
    'normalize_value1'  : float(config.get('Dataset.real', 'Dataset.real.normalize.value1')),
    'normalize_value2'  : float(config.get('Dataset.real', 'Dataset.real.normalize.value2')), 
    'use_subset'        : bool(config.getboolean('Dataset.real', 'Dataset.real.use_subset')),
} 

option_dataset_fake = {
    'channel'       : int(config.get('Dataset.fake', 'Dataset.fake.channel')),
    'height'        : int(config.get('Dataset.fake', 'Dataset.fake.height')),
    'width'         : int(config.get('Dataset.fake', 'Dataset.fake.width')),
    'size_latent'   : int(config.get('Dataset.fake', 'Dataset.fake.size_latent')),
} 

option_network = {

    'generator'     : config.get('Network', 'Network.generator'),
    'discriminator' : config.get('Network', 'Network.discriminator'),
    'size_feature'  : int(config.get('Network', 'Network.size_feature')),
}

option_optimize = {
    'algorithm'     : config.get('Optimize', 'Optimize.algorithm'),
    'size_batch'    : int(config.get('Optimize', 'Optimize.size_batch')),
    'number_epoch'  : int(config.get('Optimize', 'Optimize.number_epoch')),
}

option_optimize_generator = {
    'learning_rate' : float(config.get('Optimize.generator', 'Optimize.generator.learning_rate')),
    'loss_fidelity' : config.get('Optimize.generator', 'Optimize.generator.loss_fidelity'),
    'loss_regular'  : config.get('Optimize.generator', 'Optimize.generator.loss_regular'),
    'weight_regular': float(config.get('Optimize.generator', 'Optimize.generator.weight_regular')),
    'momentum'      : float(config.get('Optimize.generator', 'Optimize.generator.momentum')),
}

option_optimize_discriminator = {
    'learning_rate'         : float(config.get('Optimize.discriminator', 'Optimize.discriminator.learning_rate')),
    'loss'                  : config.get('Optimize.discriminator', 'Optimize.discriminator.loss'),
    'weight_gradient_decay' : float(config.get('Optimize.discriminator', 'Optimize.discriminator.weight_gradient_decay')),
    'momentum'              : float(config.get('Optimize.discriminator', 'Optimize.discriminator.momentum')),
}

option_optimize_fake = {
    'learning_rate' : float(config.get('Optimize.fake', 'Optimize.fake.learning_rate')),
    'loss_fidelity' : config.get('Optimize.fake', 'Optimize.fake.loss_fidelity'),
    'loss_regular'  : config.get('Optimize.fake', 'Optimize.fake.loss_regular'),
    'weight_regular': float(config.get('Optimize.fake', 'Optimize.fake.weight_regular')),
}

option_result = {
    'save'              : bool(config.getboolean('Result', 'Result.save')),
    'save_model'        : bool(config.getboolean('Result', 'Result.save_model')),
    'dir_result'        : config.get('Result', 'Result.dir_result'),
    'dir_save_figure'   : config.get('Result', 'Result.dir_save_figure'),
    'dir_save_log'      : config.get('Result', 'Result.dir_save_log'),
    'dir_save_model'    : config.get('Result', 'Result.dir_save_model'),
    'dir_save_option'   : config.get('Result', 'Result.dir_save_option'),
}

# ======================================================================
# cuda
# ======================================================================
device = torch.device(f'cuda:{cuda_device_number}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('device :', device)

# ======================================================================
# filename
# ======================================================================
now         = datetime.datetime.now()
date_stamp  = now.strftime('%Y_%m_%d') 
time_stamp  = now.strftime('%H_%M_%S') 

file_figure_loss                = '{},loss.png'.format(time_stamp)
file_figure_prediction          = '{},prediction.png'.format(time_stamp)
file_figure_image_train         = '{},image,train.png'.format(time_stamp)
file_figure_image_update_train  = '{},image,update,train.png'.format(time_stamp)
file_figure_image_test          = '{},image,test.png'.format(time_stamp)
file_figure_image_real          = '{},image,real.png'.format(time_stamp)
file_log                        = '{}.log'.format(time_stamp)
file_option                     = '{}.ini'.format(time_stamp)

dir_result      = os.path.join(option_result['dir_result'], '{}/'.format(option_dataset_real['name']))
dir_result_date = os.path.join(option_result['dir_result'], '{}/{}/'.format(option_dataset_real['name'], date_stamp))

path_figure     = os.path.join(dir_result_date, option_result['dir_save_figure'])
path_log        = os.path.join(dir_result_date, option_result['dir_save_log'])
path_model      = os.path.join(dir_result_date, option_result['dir_save_model'])
path_option     = os.path.join(dir_result_date, option_result['dir_save_option'])

file_figure_loss                = os.path.join(path_figure, file_figure_loss)
file_figure_prediction          = os.path.join(path_figure, file_figure_prediction)
file_figure_image_train         = os.path.join(path_figure, file_figure_image_train)
file_figure_image_update_train  = os.path.join(path_figure, file_figure_image_update_train)
file_figure_image_test          = os.path.join(path_figure, file_figure_image_test)
file_figure_image_real          = os.path.join(path_figure, file_figure_image_real)
file_log                        = os.path.join(path_log, file_log)
file_option                     = os.path.join(path_option, file_option)

if not os.path.exists(dir_result):
    os.makedirs(dir_result)
    pass 

if not os.path.exists(dir_result_date):
    os.makedirs(dir_result_date)
    pass 

if not os.path.exists(path_figure):
    os.mkdir(path_figure)
    pass 

if not os.path.exists(path_log):
    os.mkdir(path_log)
    pass 

if not os.path.exists(path_model):
    os.mkdir(path_model)
    pass 

if not os.path.exists(path_option):
    os.mkdir(path_option)
    pass 

# ======================================================================
# size of the data
# ======================================================================
size_real   = {'channel' : option_dataset_real['channel'], 'height' : option_dataset_real['height'], 'width' : option_dataset_real['width']}
size_fake   = {'channel' : option_dataset_fake['channel'], 'height' : option_dataset_fake['height'], 'width' : option_dataset_fake['width']}

# ======================================================================
# neural network model
# ======================================================================
dim_feature                 = option_network['size_feature']
size_latent                 = option_dataset_fake['size_latent']
dim_generator_input         = [size_latent, 1, 1]
dim_generator_output        = [size_fake['channel'], size_fake['height'], size_fake['width']]
dim_discriminator_input     = [size_real['channel'], size_real['height'], size_real['width']]
dim_discriminator_output    = [1, 1, 1]

generator       = Network_Generator(dim_generator_input, dim_generator_output, dim_feature, option_network['generator']).to(device)
discriminator   = Network_Discriminator(dim_discriminator_input, dim_discriminator_output, dim_feature, option_network['discriminator']).to(device) 

# ======================================================================
# dataset
# ======================================================================
path_directory  = option_dataset_real['directory']
name_dataset    = option_dataset_real['name'].lower() 
path_dataset    = os.path.join(path_directory, name_dataset)

transform_list = []
transform_list.append(transforms.ToTensor())
transform_list.append(transforms.Resize((size_real['height'], size_real['width'])))

if 'MNIST'.lower() in name_dataset:

    transform_compose   = transforms.Compose(transform_list)
    dataset_real        = MNIST(path_directory, download=False, train=True, transform=transform_compose)
    pass

elif 'CelebA'.lower() in name_dataset:
    
    transform_list.append(transforms.CenterCrop((size_real['height'], size_real['width'])))
    transform_compose   = transforms.Compose(transform_list)
    #dataset_real        = CelebA(path_dataset, download=False, transform=transform_compose, split='train')
    dataset_real        = ImageFolder(path_dataset, transform=transform_compose)
    pass

elif name_dataset == 'ImageNet'.lower():

    transform_compose   = transforms.Compose(transform_list)
    dataset_real        = ImageNet(path_dataset, download=True, transform=transform_compose, split='train')
    pass

else:
    pass

# ======================================================================
# subset of the dataset
# ======================================================================
use_subset = option_dataset_real['use_subset']

if use_subset:

    if ('MNIST'.lower() in name_dataset):
        label_real          = 4
        idx_label_real      = (dataset_real.targets == label_real)
        dataset_real.data   = dataset_real.data[idx_label_real]

# ======================================================================
# real dataset 
# ======================================================================
dataloader_real     = DataLoader(dataset=dataset_real, batch_size=1, drop_last=False, shuffle=False)
number_data_real    = len(dataset_real)
dim_data_real       = [number_data_real, size_real['channel'], size_real['height'], size_real['width']]
data_real           = torch.zeros(dim_data_real)

for i, (batch_data_real, batch_label_real) in enumerate(dataloader_real):
    value_min       = option_dataset_real['normalize_value1']
    value_max       = option_dataset_real['normalize_value2'] 
    batch_data_real = normalize_tensor(batch_data_real, value_min, value_max)
    data_real[i]    = batch_data_real

# ======================================================================
# iteration for the mini-batch of real data
# ======================================================================
size_batch                  = option_optimize['size_batch'] 
number_batch_real           = np.floor(number_data_real / size_batch).astype(int)
number_data_epoch_real      = number_batch_real * size_batch 
number_data_epoch_real_last = number_data_real - number_data_epoch_real

index_data_real             = np.arange(0, number_data_real) 
index_data_epoch_real       = np.arange(0, number_data_epoch_real) 
index_data_epoch_real_last  = np.arange(number_data_epoch_real, number_data_real)

dim_batch_data_real         = [size_batch, size_real['channel'], size_real['height'], size_real['width']]

print('dim of real data: ', dim_data_real)
print('dim of real data batch: ', dim_batch_data_real)

# ======================================================================
# optimizers
# ======================================================================
number_epoch = option_optimize['number_epoch']

if option_optimize['algorithm'].lower() == 'SGD'.lower():
    optimizer_discriminator = torch.optim.SGD(discriminator.parameters(), lr=option_optimize_discriminator['learning_rate'], momentum=option_optimize_discriminator['momentum'])
    optimizer_generator     = torch.optim.SGD(generator.parameters(), lr=option_optimize_generator['learning_rate'], momentum=option_optimize_generator['momentum'])
    
    #lr_gamma = 0.99
    #scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator, number_epoch)
    #scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(optimizer_discriminator, gamma=lr_gamma)
    #scheduler_generator     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, number_epoch)
    #scheduler_generator     = torch.optim.lr_scheduler.ExponentialLR(optimizer_generator, gamma=lr_gamma)
    
elif option_optimize['algorithm'].lower() == 'Adam'.lower():
    # needs to be modified with approprimate parameters
    beta1 = 0.5
    beta2 = 0.999
    
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=option_optimize_discriminator['learning_rate'], betas=(beta1,beta2))
    optimizer_generator     = torch.optim.Adam(generator.parameters(), lr=option_optimize_generator['learning_rate'], betas=(beta1,beta2))

else:
    pass

# ======================================================================
# for saving results
# ======================================================================
figure_image_train          = plt.figure()
figure_image_update_train   = plt.figure()
figure_image_test           = plt.figure()
figure_image_real           = plt.figure()
figure_curve                = plt.figure()
figure_prediction           = plt.figure()

logger = Logger(file_log)

with open(file_option, 'w') as configfile:    # save
    config.write(configfile)

value_loss_total_mean           = np.zeros(number_epoch)
value_loss_generator_mean       = np.zeros(number_epoch)
value_loss_generator_std        = np.zeros(number_epoch)
value_loss_discriminator_mean   = np.zeros(number_epoch)
value_loss_discriminator_std    = np.zeros(number_epoch)
value_loss_fake_mean            = np.zeros(number_epoch)
value_loss_fake_std             = np.zeros(number_epoch)

value_prediction_real_mean      = np.zeros(number_epoch)
value_prediction_real_std       = np.zeros(number_epoch)
value_prediction_fake_mean      = np.zeros(number_epoch)
value_prediction_fake_std       = np.zeros(number_epoch)
value_prediction_sum            = np.zeros(number_epoch)


# ---------------------------------------------------------------------------
#  
# function for computing the gradient penalty
#  
# ---------------------------------------------------------------------------
def compute_gradient_norm(input_data, output_prediction):
   
    number_data     = len(input_data)
    gradient        = torch.autograd.grad(outputs=output_prediction.sum(), inputs=input_data, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_square = gradient.pow(2)
    gradient_norm   = gradient_square.reshape(number_data, -1).sum(1) 
   
    return gradient_norm

# ---------------------------------------------------------------------------
#  
# function for computing the loss of the discriminator
#  
# ---------------------------------------------------------------------------
def compute_loss_discriminator(discriminator, data_fake, data_real, option):

    prediction_real = discriminator(data_real.requires_grad_())
    prediction_fake = discriminator(data_fake.requires_grad_())

    # print('type of prediction of real data : ', prediction_real.type())
    # print('type of prediction of fake data : ', prediction_fake.type())

    if option['loss'] == 'cross_entropy':
        criterion   = nn.BCEWithLogitsLoss()
        label_real  = torch.ones_like(prediction_real)
        label_fake  = torch.zeros_like(prediction_fake)        
        loss_real   = criterion(prediction_real, label_real)
        loss_fake   = criterion(prediction_fake, label_fake)
        
        loss        = (loss_real + loss_fake) / 2.0
        
    elif option['loss'] == 'wasserstein':
        sig         = nn.Sigmoid() 
        
        loss        = - torch.mean(sig(prediction_real)) + torch.mean(sig(prediction_fake))
        #loss        = - torch.mean(prediction_real) + torch.mean(prediction_fake)

    elif option['loss'] == 'l2':
        criterion   = nn.MSELoss()
        sig         = nn.Sigmoid() 
        label_real  = torch.ones_like(prediction_real)
        label_fake  = torch.zeros_like(prediction_fake)        
        loss_real   = criterion(sig(prediction_real), label_real)
        loss_fake   = criterion(sig(prediction_fake), label_fake)
        
        loss        = (loss_real + loss_fake) / 2.0

    else:
        loss        = None

    gradient_norm_real          = compute_gradient_norm(data_real, prediction_real)
    gradient_norm_fake          = compute_gradient_norm(data_fake, prediction_fake)
    loss_gradient_decay_real    = option['weight_gradient_decay'] * gradient_norm_real.mean()
    loss_gradient_decay_fake    = option['weight_gradient_decay'] * gradient_norm_fake.mean()
    
    loss = loss + loss_gradient_decay_real + loss_gradient_decay_fake 

    # the following code is applied after the update with wasserstein loss
    #if option_loss_discriminator == 'wasserstein':
    #    threshold = 1 
    #
    #    for p in discriminator.parameters():
    #        p.data.clamp_(-threshold, threshold)

    return loss, prediction_real, prediction_fake


# ---------------------------------------------------------------------------
#  
# function for computing the loss of the fake distribution
#  
# ---------------------------------------------------------------------------
def compute_loss_fake(discriminator, data_fake, option):

    prediction_fake = discriminator(data_fake)

    # ---------------------------------------------------------------------------
    # data fidelity
    # ---------------------------------------------------------------------------
    if option['loss_fidelity'] == 'cross_entropy':
        #criterion       = nn.BCEWithLogitsLoss(reduction='sum')
        criterion       = nn.BCEWithLogitsLoss()
        label_real      = torch.ones_like(prediction_fake)
        loss_fidelity   = criterion(prediction_fake, label_real)

    elif option['loss_fidelity'] == 'wasserstein':    
        sig             = nn.Sigmoid() 
        loss_fidelity   = - torch.mean(sig(prediction_fake))
        #loss_fidelity   = - torch.mean(prediction_fake)

    elif option['loss_fidelity'] == 'l2':    
        sig             = nn.Sigmoid()
        #criterion       = nn.MSELoss(reduction='sum')
        criterion       = nn.MSELoss()
        label_real      = torch.ones_like(prediction_fake)
        loss_fidelity   = criterion(sig(prediction_fake), label_real)
        
    else:
        loss_fidelity   = None

    # ---------------------------------------------------------------------------
    # regularization 
    # ---------------------------------------------------------------------------
    if option['loss_regular'] == 'tv': 
        loss_regular = (
            torch.sum(torch.abs(data_fake[:, :, :, :-1] - data_fake[:, :, :, 1:])) + 
            torch.sum(torch.abs(data_fake[:, :, :-1, :] - data_fake[:, :, 1:, :]))
            )
        
    elif option['loss_regular'] == 'l2': 
        loss_regular = (
            torch.sum(torch.square(data_fake[:, :, :, :-1] - data_fake[:, :, :, 1:])) + 
            torch.sum(torch.square(data_fake[:, :, :-1, :] - data_fake[:, :, 1:, :]))
            ) 

    else:
        loss_regular = None 

    # ---------------------------------------------------------------------------
    # total loss
    # ---------------------------------------------------------------------------
    loss = loss_fidelity + option['weight_regular'] * loss_regular
    
    return loss 


# ---------------------------------------------------------------------------
#  
# function for computing the loss of the generator
#  
# ---------------------------------------------------------------------------
def compute_loss_generator(data_fake, data_fake_update, option):
       
    if option['loss_fidelity'] == 'l2':
        criterion       = nn.MSELoss()
        loss_fidelity   = criterion(data_fake, data_fake_update)

    elif option['loss_fidelity'] == 'l1':
        criterion       = nn.L1Loss()
        loss_fidelity   = criterion(data_fake, data_fake_update)

    elif option['loss_fidelity'] == 'cross_entropy':
        criterion       = nn.KLDivLoss(reduction = 'batchmean')
        loss_fidelity   = criterion(data_fake, data_fake_update)
                
    else:
        loss_fidelity = None
       
    # ---------------------------------------------------------------------------
    # regularization 
    # ---------------------------------------------------------------------------
    if option['loss_regular'] == 'tv': 
        loss_regular = (
            torch.sum(torch.abs(data_fake[:, :, :, :-1] - data_fake[:, :, :, 1:])) + 
            torch.sum(torch.abs(data_fake[:, :, :-1, :] - data_fake[:, :, 1:, :]))
            )
        
    elif option['loss_regular'] == 'l2': 
        loss_regular = (
            torch.sum(torch.square(data_fake[:, :, :, :-1] - data_fake[:, :, :, 1:])) + 
            torch.sum(torch.square(data_fake[:, :, :-1, :] - data_fake[:, :, 1:, :]))
            ) 

    else:
        loss_regular = None 
        
    # ---------------------------------------------------------------------------
    # total loss
    # ---------------------------------------------------------------------------
    loss = loss_fidelity + option['weight_regular'] * loss_regular
     
    return loss


# ---------------------------------------------------------------------------
#  
# function for updating discriminator
#  
# ---------------------------------------------------------------------------
def update_discriminator(discriminator, optimizer, data_fake, data_real, option):

    discriminator.train()
    optimizer.zero_grad()

    (loss_discriminator, prediction_real, prediction_fake) = compute_loss_discriminator(discriminator, data_fake, data_real, option)
    loss_discriminator.backward(retain_graph=True)
    optimizer.step()

    value_loss_discriminator    = loss_discriminator.item()
    value_prediction_real       = prediction_real.detach().cpu()
    value_prediction_fake       = prediction_fake.detach().cpu()

    return (value_loss_discriminator, value_prediction_real, value_prediction_fake)


# ---------------------------------------------------------------------------
#  
# function for updating generator
#  
# ---------------------------------------------------------------------------
def update_fake(discriminator, variable_fake, option):

    discriminator.eval()
   
    if variable_fake.grad is not None:
        variable_fake.grad.zero_()

    loss_fake = compute_loss_fake(discriminator, variable_fake, option)
    loss_fake.backward()
    
    variable_fake.data = variable_fake.data - option['learning_rate'] * variable_fake.grad
        
    #print('max gradient: ', variable_fake.grad.max())
    #print('min gradient: ', variable_fake.grad.min())

    #print('max update: ', variable_fake.data.max())
    #print('min update: ', variable_fake.data.min())
    
    #value_loss_fake = loss_fake.item() / len(variable_fake)
    value_loss_fake = loss_fake.item()

    return value_loss_fake


# ---------------------------------------------------------------------------
#  
# function for updating generator
#  
# ---------------------------------------------------------------------------
def update_generator(generator, optimizer, data_fake, data_fake_update, option):

    generator.train()
    optimizer.zero_grad()
    
    loss_generator = compute_loss_generator(data_fake, data_fake_update, option)
    loss_generator.backward()
    optimizer.step()
   
    value_loss_generator = loss_generator.item()
    
    return value_loss_generator


# ---------------------------------------------------------------------------
#  
# main iterations (epoch)
#  
# ---------------------------------------------------------------------------
for epoch in range(number_epoch):
    
    index_batch_real_shuffle    = np.random.permutation(index_data_real) 
    index_batch_real_drop_last  = index_batch_real_shuffle[0:number_data_epoch_real] 
    index_batch_real_slice      = np.reshape(index_batch_real_drop_last, [number_batch_real, size_batch]) 

    value_loss_batch_average        = []
    value_loss_batch_generator      = []
    value_loss_batch_discriminator  = []
    value_loss_batch_fake           = []

    value_prediction_batch_real     = []
    value_prediction_batch_fake     = []
    value_prediction_batch_average  = []
        
        
    # ---------------------------------------------------------------------------
    #  
    # mini-batch iterations for real data
    #  
    # ---------------------------------------------------------------------------
    
    for i in tqdm(range(number_batch_real)):

        batch_data_real = data_real[index_batch_real_slice[i]]
        batch_data_real = batch_data_real.to(device)
       
        # ---------------------------------------------------------------------------
        #  
        # update fake data
        #  
        # ---------------------------------------------------------------------------
        batch_data_latent   = torch.randn(size_batch, size_latent).to(device) 
        batch_data_latent   = torch.reshape(batch_data_latent, [size_batch, size_latent, 1, 1])
        batch_data_fake     = generator(batch_data_latent)

        # ---------------------------------------------------------------------------
        #  
        # update the discriminator
        #  
        # ---------------------------------------------------------------------------
        (value_loss_discriminator, value_prediction_real, value_prediction_fake) = update_discriminator(discriminator, optimizer_discriminator, batch_data_fake, batch_data_real, option_optimize_discriminator)

        # ---------------------------------------------------------------------------
        #  
        # refresh latent for the fake and the generator
        #  
        # ---------------------------------------------------------------------------
        use_refresh_latent = FALSE
        
        if use_refresh_latent == True:
            batch_data_latent   = torch.randn(size_batch, size_latent).to(device) 
            batch_data_latent   = torch.reshape(batch_data_latent, [size_batch, size_latent, 1, 1])
            batch_data_fake     = generator(batch_data_latent)
        
        # ---------------------------------------------------------------------------
        #  
        # update the fake
        #  
        # ---------------------------------------------------------------------------
        batch_variable_fake     = Parameter(batch_data_fake, requires_grad=True)
        value_loss_fake         = update_fake(discriminator, batch_variable_fake, option_optimize_fake) 
        batch_data_fake_update  = batch_variable_fake.data
        
        # ---------------------------------------------------------------------------
        #  
        # update the generator
        #  
        # ---------------------------------------------------------------------------        
        value_loss_generator = update_generator(generator, optimizer_generator, batch_data_fake, batch_data_fake_update, option_optimize_generator)
         
        # ---------------------------------------------------------------------------
        #  
        # save loss and prediction for each real data batch 
        #  
        # ---------------------------------------------------------------------------
        value_prediction_batch_real     = np.append(value_prediction_batch_real, value_prediction_real)
        value_prediction_batch_fake     = np.append(value_prediction_batch_fake, value_prediction_fake)
        value_prediction_batch_average  = np.append(value_prediction_batch_average, (value_prediction_real + value_prediction_fake) * 0.5)
        
        value_loss_batch_discriminator  = np.append(value_loss_batch_discriminator, value_loss_discriminator)
        value_loss_batch_generator      = np.append(value_loss_batch_generator, value_loss_generator)
        value_loss_batch_fake           = np.append(value_loss_batch_fake, value_loss_fake)
        value_loss_batch_average        = np.append(value_loss_batch_average, (value_loss_batch_discriminator + value_loss_batch_fake) * 0.5) 


    # ---------------------------------------------------------------------------
    #  
    # schedule the learning rate
    #  
    # ---------------------------------------------------------------------------
    '''
    if option_optimize['algorithm'].lower() == 'SGD'.lower():
        scheduler_discriminator.step() 
        scheduler_generator.step() 
    '''
    
    # ---------------------------------------------------------------------------
    #  
    # save results 
    #  
    # ---------------------------------------------------------------------------
    value_loss_discriminator_mean[epoch]    = np.mean(value_loss_batch_discriminator)
    value_loss_discriminator_std[epoch]     = np.std(value_loss_batch_discriminator)
    value_loss_generator_mean[epoch]        = np.mean(value_loss_batch_generator)
    value_loss_generator_std[epoch]         = np.std(value_loss_batch_generator)
    value_loss_fake_mean[epoch]             = np.mean(value_loss_batch_fake)
    value_loss_fake_std[epoch]              = np.std(value_loss_batch_fake)
    value_loss_total_mean[epoch]            = (value_loss_discriminator_mean[epoch] + value_loss_fake_mean[epoch]) * 0.5 
    
    value_prediction_real_mean[epoch]       = np.mean(value_prediction_batch_real)
    value_prediction_real_std[epoch]        = np.std(value_prediction_batch_real)
    value_prediction_fake_mean[epoch]       = np.mean(value_prediction_batch_fake)
    value_prediction_fake_std[epoch]        = np.std(value_prediction_batch_fake)
    value_prediction_sum[epoch]             = (value_prediction_real_mean[epoch] + value_prediction_fake_mean[epoch]) * 0.5

    figure_image_train.clf()
    figure_image_update_train.clf()
    figure_image_test.clf()
    figure_image_real.clf()
    figure_curve.clf()
    figure_prediction.clf()

    size_test       = size_batch
    data_latent     = torch.randn(size_test, size_latent).to(device) 
    data_latent     = torch.reshape(data_latent, [size_test, size_latent, 1, 1])
    data_generate   = generator(data_latent)
   
    image_fake_train            = batch_data_fake.data
    image_fake_update_train     = batch_data_fake_update
    image_fake_test             = data_generate
    image_real                  = batch_data_real
    
    plot_image_grid(figure_image_train, image_fake_train, file_figure_image_train)
    plot_image_grid(figure_image_update_train, image_fake_update_train, file_figure_image_update_train)
    plot_image_grid(figure_image_test, image_fake_test, file_figure_image_test)
    plot_image_grid(figure_image_real, image_real, file_figure_image_real)
    
    plot_curve_errorbar3(figure_curve, value_loss_generator_mean, value_loss_generator_std, 'generator', value_loss_discriminator_mean, value_loss_discriminator_std, 'discriminator',  value_loss_fake_mean, value_loss_fake_std, 'fake', file_figure_loss)
    plot_curve_errorbar3(figure_prediction, value_prediction_real_mean, value_prediction_real_std, 'real', value_prediction_fake_mean, value_prediction_fake_std, 'fake',  value_prediction_sum, 0, 'sum', file_figure_prediction)
    
    message = '[%04d/%04d] (G) %10.7f, (D) %10.7f, (F) %10.7f, (D+F) %10.7f\n' % (epoch, number_epoch, value_loss_generator_mean[epoch], value_loss_discriminator_mean[epoch],  value_loss_fake_mean[epoch], value_loss_total_mean[epoch])
    logger.write(message)

    # save models at each epoch
    if option_result['save_model']:
        path_file_model = os.path.join(path_model, time_stamp)

        if not os.path.exists(path_file_model):            
            os.mkdir(path_file_model)

        file_model  = '{:05d}.pth'.format(epoch)
        file_model  = os.path.join(path_file_model, file_model)
        discriminator.save(file_model)
        pass

plt.close(figure_image_train)
plt.close(figure_image_update_train)
plt.close(figure_image_test)
plt.close(figure_image_real)

plt.close(figure_curve)
plt.close(figure_prediction)

logger.close()

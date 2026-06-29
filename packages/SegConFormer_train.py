## This code was written by Anes Ouadou

import os
import sys
import pathlib
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.functional import relu

import torchvision 
from torchvision import transforms, models
from torchvision.io import read_image 

import torch.optim as optim
from pytorch_optimizer import Lamb

import matplotlib.pyplot as plt
import time
import copy
import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score

import pandas as pd
from statistics import mean 
import argparse

import stemming as st
import AtrousSpatialPyramidPooling as aspp
import ConvBlockAttModule as cbam
import trblock as trb
import UpsampleModule as upsample
import fusion
import decoder
import dataset
import segconformer 

'''
#######################  DEFINE ARGUMENTS ########################################################

parser = argparse.ArgumentParser()

parser.add_argument('--train','-t',
                    help     = 'The path to the list of training images',
                    metavar  = '',
                    required = True)

parser.add_argument('--val','-v',
                    help     = 'The path to the list of validation images',
                    metavar  = '',
                    required = True)

parser.add_argument('--epochs','-e',
                    help     = 'The number of epochs the model is trained',
                    metavar  = '',
                    type     = int,
                    default  = 50,
                    required = True)

parser.add_argument('--batch_size','-b',
                    help     = 'The batch size',
                    metavar  ='',
                    type     = int,
                    default  = 2,
                    required = True)

parser.add_argument('--workers','-w',
                    help     = 'The number of workers per GPU',
                    metavar  ='',
                    type     = int,
                    default  = 2,
                    required = True)
                 
parser.add_argument('--log',
                    help    = 'text file that stores the status of each image',
                    default = 'train_metrics.log')         

args = parser.parse_args()

#####################    GET ARGUMENTS         ######################################################


n_epochs             = args.epochs
batch_size           = args.batch_size
workers              = args.workers
train_list_path      = args.train
validation_list_path = args.val
'''


branch1 = {'in_channels':64,  # output size 256
           'out_channels':64,
           'kernel_size':3,
           'dilation':1,
           'stride':1,
           'padding':1}

branch2 = {'in_channels':64, # output size 128
           'out_channels':64,
           'kernel_size':27,
           'dilation':5,
           'stride':1,
           'padding':1}

branch3 = {'in_channels':64, # output size  64
           'out_channels':64,
           'kernel_size':35,
           'dilation':2,
           'stride':3,
           'padding':1}

branch4 = {'in_channels':64, # output size  32
           'out_channels':64,
           'kernel_size':11,
           'dilation':4,
           'stride':7,
           'padding':1}

aspp_branchs = {'branch1':branch1,
                'branch2':branch2,
                'branch3':branch3,
                'branch4':branch4}


#####################    custom dataset class  ######################################################

# torch.nn is the module used for building neural networks  
# torchvision allows us to import pre-trained models 

# # Create data

# The data used with Pytorch needs to be in tensor format.  
# There are different ways to convert data into **tensor** format depending on the source data.  
# - using torch.tensor:
#     data can be converted using *torch.tensor* the data type is automatically inferred    
#     x_data = torch.tensor(x_source)
#       
# - using from_numpy: if the source data is of type numpy, then we can use from_numpy method.    
#     
#     x_data = torch.from_numpy(x_numpy)


# dataset definition

# path to images
images_folder = 'training_data_png/'
images_type   = 'tcis'

# path to labels
labels_folder = 'chips_labels'

## train dataset
traindataset = dataset.buildingdataset(image_folder=images_folder,
                               label_folder=labels_folder, 
                               image_type=images_type, 
                               train = True,
                               img_extension='.png',
                               lab_extension='.npy')

## train dataloader
batch_size  = 1
workers     = 0
train_dl = DataLoader(traindataset, 
                      batch_size=batch_size,
                      num_workers = workers,
                      shuffle=True)


## validation dataset
validatedataset = dataset.buildingdataset(image_folder=images_folder,
                               label_folder=labels_folder, 
                               image_type=images_type, 
                               train = False,
                               img_extension='.png',
                               lab_extension='.npy')

## validation dataloader
batch_size  = 1
workers     = 0
validate_dl = DataLoader(validatedataset, 
                      batch_size=batch_size,
                      num_workers = workers,
                      shuffle=True)


#####################    # # Build a U-Net class   ######################################################


# We create a class for our U-Net model  
# This class inherits from the nn.Module  
# The argument **n_class** specifies the number of classes in the label mask
# 

# ## Forward method

# The forward method dictates how data is processed as it passes through the network, first through the encoder part, then the decoder part, and finally through the out layer where the pixel classification is performed. 

model = segconformer.segconformer(aspp_opt=aspp_branchs,n_class=1)

print(model)


#####################    Logging key information      ######################################################

logging.basicConfig(filename="U-Net_training_{}.log".format(fold), filemode='w', level=logging.DEBUG)
logging.debug('Reading arguments')
print('Reading arguments')
'''
logging.debug("U_Net")
logging.debug('Fold number {}'.format(fold))
logging.debug('Number of train samples {}'.format(len(train_list)))
logging.debug('Number of validation samples {}'.format(len(validation_list)))
logging.debug('Number of epochs {}'.format(n_epochs))
logging.debug('Batch size {}'.format(batch_size))
logging.debug('Check if GPU is available: {}'.format(torch.cuda.is_available()))
logging.debug('The total number of GPUs available: {}'.format(torch.cuda.device_count()))
logging.debug('The name of the GPU available: {}'.format(torch.cuda.get_device_name(0)))
'''

print("SegConFormer")
print('Number of train samples {}'.format(len(traindataset)))
print('Number of validation samples {}'.format(len(validatedataset)))
print('Number of epochs {}'.format(n_epochs))
print('Batch size {}'.format(batch_size))
print('Check if GPU is available: {}'.format(torch.cuda.is_available()))
print('The total number of GPUs available: {}'.format(torch.cuda.device_count()))
print('The name of the GPU available: {}'.format(torch.cuda.get_device_name(0)))

#####################    Prepare for training   ######################################################

logging.debug('preparing model and training parameters ...')
print('preparing model and training parameters ...')

model = UNet(n_class=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.0001, 
                      momentum=0.9)

total_params = sum(p.numel() for p in model.parameters())

logging.debug('The total number of parameters is: {}'.format(total_params))
print('The total number of parameters is: {}'.format(total_params))

# # Load model to GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    logging.debug("Model moved to GPU")
else:
    logging.debug("GPU is not available, using CPU instead")


epoch_list = []

train_time_per_epoch      = []
train_loss_per_epoch      = []
validation_loss_per_epoch = []

#####################    start the training   ######################################################

for epoch in range(n_epochs): # outer loop for epochs
    
    logging.debug('Train epoch: {}'.format(epoch))
    print('Train epoch: {}'.format(epoch))

    start_time = current_time = time.time()
    epoch_list.append(epoch)

    ## Train phase of the epoch ##
    model.train()
    
    iter_count = 1
    train_time_per_iteration = []
    train_loss_per_iteration = []
    
    for inputs, labels in train_dl: # inner loop for iterations (one batch at a time)
        
        device = torch.device("cuda")
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        y_pred = model(inputs)
        loss   = loss_fn(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_per_iteration.append(loss.item()) # save iteration loss
        
        new_time = time.time()
        dtime = new_time - current_time 
        
        train_time_per_iteration.append(dtime)
        
        iter_count += 1
        current_time = copy.copy(new_time)

    train_time_per_epoch.append(sum(train_time_per_iteration))
    train_loss_per_epoch.append(mean(train_loss_per_iteration)) # compute and save average training loss per repoch
    
    #print('epoch {} train time: {:.2f} average loss: {:.2f}'.format(epoch,sum(train_time_per_iteration),mean(train_loss_per_iteration)))
    logging.debug('epoch {} train time: {:.2f} average loss: {:.2f}'.format(epoch,sum(train_time_per_iteration),mean(train_loss_per_iteration)))

    logging.debug('Validation epoch: {}'.format(epoch))
    print('Validation epoch: {}'.format(epoch))
   

    ## Validation phase of the epoch ##
    model.eval()
    
    metrics     = []
    masks_truth = []
    masks_pred  = []
    val_loss_per_sample_list = []
    
    for inputs, labels in test_dl:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        y_pred  = model(inputs)
        
        val_loss_per_sample = loss_fn(y_pred, labels)
        val_loss_per_sample_list.append(val_loss_per_sample.item())
                
    val_loss = mean(val_loss_per_sample_list)
    validation_loss_per_epoch.append(val_loss)
    
    logging.debug('epoch: {} loss: {}'.format(epoch,val_loss))
    print('epoch: {} loss: {}'.format(epoch,val_loss))

    torch.save(model.state_dict(),'models/fold_{}/Unetmodel_{}.pth'.format(fold,epoch))
                                                                                        
# arrange data in  data frame
loss_metric_dict = {}
loss_metric_dict['epochs']          = epoch_list
loss_metric_dict['train_loss']      = train_loss_per_epoch 
loss_metric_dict['validation_loss'] = validation_loss_per_epoch

df_loss_metrics = pd.DataFrame.from_dict(loss_metric_dict)
print(df_loss_metrics)
                                                                                                          


logging.debug('loss metrics:')
logging.debug(df_loss_metrics)

df_loss_metrics.to_csv('loss_metrics_{}.csv'.format(fold))

logging.debug("training ended")
logging.debug("training time {} seconds".format(time.time() - start_time))

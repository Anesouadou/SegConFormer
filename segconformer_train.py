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
from torchvision import models
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.io import read_image 

import torch.optim.lr_scheduler as lr_scheduler
import torch_optimizer as optim
import pytorch_warmup as warmup


print(torch.__version__)
print(torch.version.cuda)  # CUDA version

from packages import stemming as st
from packages import AtrousSpatialPyramidPooling as aspp
from packages import ConvBlockAttModule as cbam
from packages import trblock as trb
from packages import UpsampleModule as upsample
from packages import fusion
from packages import decoder
from packages import dataset
#import segconformer as segconformer

import segconformer_S as segconformer_S
import segconformer_M as segconformer_M
import segconformer_M2 as segconformer_M2
import segconformer_M3 as segconformer_M3
import segconformer_M4 as segconformer_M4
import segconformer_L as segconformer_L

import time
from datetime import datetime
import copy
import logging
import pandas as pd
from statistics import mean 
import argparse
import yaml

#######################  DEFINE ARGUMENTS ########################################################

parser = argparse.ArgumentParser()

parser.add_argument('--architecture','-a',
                    help     = 'name of the architecture being trained',
                    choices  = ['S','M','M2','M3','M4','L'],
                    metavar  = '',
                    required = True)

parser.add_argument('--parameters','-p',
                    help     = 'The parameters of this experiment provided as a YAML (YML) file',
                    metavar  = '',
                    required = True)

parser.add_argument('--exp','-e',
                    help    = 'Experiment number used to distinguish between experiments',
                    required = True)

'''
parser.add_argument('--log',
                    help    = 'text file that logs the training parameters',
                    default = 'train_metrics.log')         
'''
args = parser.parse_args()

####################################################         GET ARGUMENTS         ################################################

architecture  = args.architecture
exp_paramters_file = args.parameters
exp_number         = args.exp


formatted_date = datetime.now().strftime("%m-%d-%y")

logging.basicConfig(filename='segconformer_{}_{}.log'.format(architecture,exp_number), filemode='w', level=logging.DEBUG)
logging.debug('segconformer_{}'.format(architecture)) # the type of architecture being used 
logging.debug(''.format(formatted_date)) # the date of the experiment for reference

####################################################  EXTRACT TRAINING PARAMETERS  ################################################

with open(exp_paramters_file) as file:
    exp_paramters = yaml.load(file, Loader=yaml.FullLoader)

options = list(exp_paramters.keys())
print(options)

arch_parameters = exp_paramters['arch_parameters']

stemming_chans = arch_parameters['stemming_chans']
aspp_branchs   = arch_parameters['aspp_options']
tr_chan        = arch_parameters['tr_channels']
method         = arch_parameters['upsample_option']
fusion_opt     = arch_parameters['fusion_option']
decoder_chans  = arch_parameters['decoder_chans']

n_class       = int(exp_paramters['n_class'])
n_epochs      = int(exp_paramters['n_epochs'])
warmup_period = int(exp_paramters['warmup_period']) # epochs

data_path     = exp_paramters['data_path']
images_folder = exp_paramters['images_folder']
img_extension = exp_paramters['image_extension']
masks_folder  = exp_paramters['mask_folder']
lab_extension = exp_paramters['mask_extension']

batch_size_train = int(exp_paramters['bs_train'])
workers_train    = int(exp_paramters['nw_train'])

batch_size_validate = int(exp_paramters['bs_validate'])
workers_validate    = int(exp_paramters['nw_validate'])

lr_value      = float(exp_paramters['lr_value'])
scheduler_opt = exp_paramters['scheduler']
step_size     = int(scheduler_opt['step_size'])
gamma         = float(scheduler_opt['gamma'])

path_models = exp_paramters['models_folder']


####################################################                               ################################################


logging.debug(''.format()) #


####################################################         BUILD THE MODEL        ################################################

if architecture == 'S':
    model = segconformer_S.segconformer_S(n_class=n_class,
                                          stemming_channels=stemming_chans,
                                          aspp_opt=aspp_branchs,
                                          tr_channels=tr_chan,
                                          fusion_opt=fusion_opt,
                                          decoder_channels=decoder_chans,
                                          method = method)
elif architecture == 'M':
    model = segconformer_M.segconformer_M(n_class=n_class,
                                          stemming_channels=stemming_chans,
                                          aspp_opt=aspp_branchs,
                                          tr_channels=tr_chan,
                                          fusion_opt=fusion_opt,
                                          decoder_channels=decoder_chans,
                                          method = method)
elif architecture == 'M2':
    model = segconformer_M2.segconformer_M2(n_class=n_class,
                                          stemming_channels=stemming_chans,
                                          aspp_opt=aspp_branchs,
                                          tr_channels=tr_chan,
                                          fusion_opt=fusion_opt,
                                          decoder_channels=decoder_chans,
                                          method = method)
elif architecture == 'M3':
    model = segconformer_M3.segconformer_M3(n_class=n_class,
                                          stemming_channels=stemming_chans,
                                          aspp_opt=aspp_branchs,
                                          tr_channels=tr_chan,
                                          fusion_opt=fusion_opt,
                                          decoder_channels=decoder_chans,
                                          method = method)
elif architecture == 'M4':
    model = segconformer_M4.segconformer_M4(n_class=n_class,
                                          stemming_channels=stemming_chans,
                                          aspp_opt=aspp_branchs,
                                          tr_channels=tr_chan,
                                          fusion_opt=fusion_opt,
                                          decoder_channels=decoder_chans,
                                          method = method)
elif architecture == 'L':
    model = segconformer_L.segconformer_L(n_class=n_class,
                                          stemming_channels=stemming_chans,
                                          aspp_opt=aspp_branchs,
                                          tr_channels=tr_chan,
                                          fusion_opt=fusion_opt,
                                          decoder_channels=decoder_chans,
                                          method = method)
else:
    print('The architecture selected {} does not exist'.format(architecture))
    print("You selected a non existing architecture: you need to select from this list ['S','M','M2','L']")
    sys.exit("Error: architecture error ")  
    
total_params = sum(p.numel() for p in model.parameters())
logging.debug('{:,} parameters'.format(total_params)) #
logging.debug(model)

####################################################         Data preparation        ################################################

logging.debug('image path: {}'.format(data_path))

# path to labels
#masks = 'chips_labels'
'''
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),    # Random vertical flip
    transforms.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),  # Rotate 90 degrees
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),  # Random crop and resize
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR)  # Random translation by 10%
])
'''

traindataset = dataset.buildingdataset(data_path=data_path,
                                       image_folder=images_folder,
                                       label_folder=masks_folder, 
                                       data = 'train',
                                       img_extension=img_extension,
                                       lab_extension=lab_extension)
                                       #transform=train_transform)

train_dl = DataLoader(traindataset, 
                      batch_size=batch_size_train,
                      num_workers = workers_train,
                      shuffle=True)

validatedataset = dataset.buildingdataset(data_path=data_path,
                                          image_folder=images_folder,
                                          label_folder=masks_folder, 
                                          data = 'validate',
                                          img_extension=img_extension,
                                          lab_extension=lab_extension)

validate_dl = DataLoader(validatedataset, 
                         batch_size=batch_size_validate,
                         num_workers = workers_validate,
                         shuffle=True)

# Training parameters

loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Lamb(model.parameters(),
                       lr= lr_value,
                       betas=(0.9, 0.999),
                       eps=1e-8,weight_decay=0)

def lr_lambda(epoch):
    if epoch < 50:
        return 1
    else:
        return 1 / (5 ** ((epoch - 50) // 25 + 1))


'''        
scheduler = lr_scheduler.StepLR(optimizer,
                                step_size=step_size, 
                                gamma=gamma)
'''

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
lr_per_epoch              = []

# Training and validation

training_start_time = time.time()


for epoch in range(n_epochs): # outer loop for epochs
    print('Train epoch: {}'.format(epoch))
    logging.debug('Train epoch: {}'.format(epoch))

    start_time = current_time = time.time()
    epoch_list.append(epoch)

    ## Train phase of the epoch ##
    model.train()
    
    
    train_time_per_iteration = []
    train_loss_per_iteration = []
    lr_per_iteration         = []
    
    #for inputs, labels in train_dl: # inner loop for iterations (one batch at a time)
    for batch_idx, (inputs, labels) in enumerate(train_dl): # inner loop for iterations (one batch at a time)
        
        device = torch.device("cuda")
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        y_pred = model(inputs)
        
        loss = loss_fn(y_pred,labels.float())
        
        loss.backward()
        optimizer.step()    
        
        train_loss_per_iteration.append(loss.item()) # save iteration loss
        
        new_time = time.time()
        dtime = new_time - current_time 
        
        train_time_per_iteration.append(dtime)
        
        lr_iter = optimizer.param_groups[0]['lr']
        lr_per_iteration.append(lr_iter)
        
        current_time = copy.copy(new_time)

    scheduler.step()   
    
    train_time_per_epoch.append(sum(train_time_per_iteration))
    train_loss_per_epoch.append(mean(train_loss_per_iteration)) # compute and save average training loss per repoch
    lr_per_epoch.append(mean(lr_per_iteration))

    #torch.save(model.state_dict(), '{}/weights_only/model_weights.pth'.format(path_models))
    torch.save(model, '{}/segconformer_{}/experiment{}/model_{}.pth'.format(path_models,architecture,exp_number,epoch))
    
    ## validation phase of the epoch ##
    model.eval()
    
    validation_loss_per_sample = []
    
    with torch.no_grad():
        for inputs, labels in validate_dl: # inner loop for iterations (one batch at a time)
            device = torch.device("cuda")
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)  # Perform a forward pass to get the predictions
            val_loss = loss_fn(outputs, labels.float())
            validation_loss_per_sample.append(val_loss.item()) 
    
    validation_loss_per_epoch.append(mean(validation_loss_per_sample))    

torch.save(model, '{}/segconformer_{}/experiment{}/model_last.pth'.format(path_models,architecture,exp_number))

training_end_time = time.time()
total_training_time = training_end_time - training_start_time
total_training_time

with open('{}/segconformer_{}/train_summary_{}.txt'.format(path_models,architecture,exp_number),'w') as file:
    file.write('SegConformer_{}\n'.format(architecture))
    file.write('number of parameters: {:,}\n'.format(total_params))
    file.write('training time: {}\n'.format(total_training_time))
    file.close()

train_metrics = {}
train_metrics['epochs']          = list(range(n_epochs))
train_metrics['lr']              = lr_per_epoch
train_metrics['train_loss']      = train_loss_per_epoch
train_metrics['validation_loss'] = validation_loss_per_epoch
train_metrics['epoch_time']      = train_time_per_epoch

df_train_metrics = pd.DataFrame(train_metrics)

df_train_metrics.to_csv('{}/segconformer_{}/train_summary_{}.csv'.format(path_models,architecture,exp_number),index=False)

## This code was written by Anes Ouadou

import os
import sys
import pathlib
import torch
import torch.optim as optim
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader

from torchvision import transforms, models

import torch.nn as nn
from torch.nn.functional import relu

from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import time
import copy

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score

import pandas as pd
from statistics import mean 
import argparse

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

parser.add_argument('--selected','-s',
                    help     = 'The model selected for the evaluation of data',
                    metavar  = '',
                    type     = int,
                    default  = 50,
                    required = True)

parser.add_argument('--eval_thresh','-r',
                    help     = 'The threshold used when evaluating predicted masks',
                    metavar  ='',
                    type     = float,
                    default  = 0.5,
                    required = True)
                   
parser.add_argument('--log',
                    help    = 'text file that stores the status of each image',
                    default = 'train_metrics.log')         

args = parser.parse_args()

#####################    GET ARGUMENTS         ######################################################

model_selected       = args.selected
eval_thrshold        = args.eval_thresh
train_list_path      = args.train
validation_list_path = args.val

name_split_train    = train_list_path.split('_')
fold_train = name_split_train[3][0]

name_split_validate = validation_list_path.split('_')
fold_validate = name_split_validate[3][0]

if fold_train == fold_validate:
    fold = int(fold_train)
else:
    print(train_list_path)
    print(validation_list_path)
    sys.exit('the train fold does not match the validation fold') 
    
#####################    GET GPU information   ######################################################

print(' Check if GPU is available: {}'.format(torch.cuda.is_available()))
print('the total number of GPUs available: {}'.format(torch.cuda.device_count()))
print('the name of the GPU available: {}'.format(torch.cuda.get_device_name(0)))

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

class buildingdataset(Dataset):
    # load the dataset
    def __init__(self, train_list, validation_list, train = True, transform=None):
        
        self.train_list      = train_list
        self.transform       = transform
        self.validation_list = validation_list
        if train == True:
            self.images_list = self.train_list
        else:
            self.images_list = self.validation_list
        
    # get the total number of images in the dataset
    def __len__(self):
        self.dataset_size = len(self.images_list)
        return self.dataset_size
         
    def __getitem__(self, idx):
        
        self.image_name = self.images_list[idx]
        self.namesplit  = self.image_name.split('_') 
        self.label_name = '{}_{}_{}_{}_{}_lab.npy'.format(self.namesplit[0],
                                                          self.namesplit[1],
                                                          self.namesplit[2],
                                                          self.namesplit[3],
                                                          self.namesplit[4])
     
        img = np.load('images/{}'.format(self.image_name))
        lab = np.load('labels/{}'.format(self.label_name))
        
        if img.shape[2] == 3:
            img = np.moveaxis(img,2,0)
        
        if self.transform:
            img = self.transform(img)
        
        img = torch.from_numpy(img)
        lab = torch.from_numpy(lab)
        lab = lab.type(torch.LongTensor)
        return [img,lab]
            
#####################    # # Build a U-Net class   ######################################################


# We create a class for our U-Net model  
# This class inhirit from the nn.Module  
# the argument **n_class** specifies the number of classes in the label mask
# 

# ## Forward method

# The forward method dictates how data is processed as it passes through the network, first through the encoder part, then the decoder part, and finally through the out layer where the pixel classification is performed. 

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 3x650x650
        self.ec11  = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 64x650x650
        self.eb11  = nn.BatchNorm2d(64)
        self.ea11  = nn.ReLU() 
        self.ec12  = nn.Conv2d(64, 64, kernel_size=3, padding=0) # output: 64x648x648
        self.eb12  = nn.BatchNorm2d(64)
        self.ea12  = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # output: 64x324x324

        # input: 64x324x324
        self.ec21  = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 128x324x324
        self.eb21  = nn.BatchNorm2d(128)
        self.ea21  = nn.ReLU() 
        self.ec22  = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 128x324x324
        self.eb22  = nn.BatchNorm2d(128)
        self.ea22  = nn.ReLU() 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)        # output: 128x162x162

        # input: 128x162x162
        self.ec31  = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 256x162x162
        self.eb31  = nn.BatchNorm2d(256)
        self.ea31  = nn.ReLU() 
        self.ec32  = nn.Conv2d(256, 256, kernel_size=3, padding=0) # output: 256x160x160
        self.eb32  = nn.BatchNorm2d(256)
        self.ea32  = nn.ReLU() 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)         # output: 256x80x80

        # input: 256x80x80
        self.ec41  = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 512x80x80
        self.eb41  = nn.BatchNorm2d(512)
        self.ea41  = nn.ReLU() 
        self.ec42  = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 512x80x80
        self.eb42  = nn.BatchNorm2d(512)
        self.ea42  = nn.ReLU() 
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)         # output: 512x40x40

        # input: 512x40x40
        self.ec51  = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output: 1024x40x40
        self.eb51   = nn.BatchNorm2d(1024)
        self.ea51  = nn.ReLU() 
        self.ec52  = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 1024x40x40
        self.eb52   = nn.BatchNorm2d(1024)
        self.ea52  = nn.ReLU() 

        # Decoder
        # input: 1024x40x40
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0) # output: 512x80x80 
        self.dc11    = nn.Conv2d(1024, 512, kernel_size=3, padding=1)                    # output: 512x80x80
        self.db11    = nn.BatchNorm2d(512)
        self.da11    = nn.ReLU() 
        self.dc12    = nn.Conv2d(512, 512, kernel_size=3, padding=1)                     # output: 512x80x80
        self.db12    = nn.BatchNorm2d(512)
        self.da12    = nn.ReLU()
        
        # input: 512x80x80
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, dilation=2, output_padding=1) # output: 256x162x162
        self.dc21    = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.db21    = nn.BatchNorm2d(256)
        self.da21    = nn.ReLU() 
        self.dc22    = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.db22    = nn.BatchNorm2d(256)
        self.da22    = nn.ReLU() 
        
        # input: 256x162x162
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0) # output: 128x324x324
        self.dc31    = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.db31    = nn.BatchNorm2d(128)
        self.da31    = nn.ReLU() 
        self.dc32    = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.db32    = nn.BatchNorm2d(128)
        self.da32    = nn.ReLU() 
        
        # input: 256x324x324
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, dilation=2, output_padding=1) # output: 64x650x650
        self.dc41    = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.db41    = nn.BatchNorm2d(64)
        self.da41    = nn.ReLU() 
        self.dc42    = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.db42    = nn.BatchNorm2d(64)
        self.da42    = nn.ReLU() 

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        xe11 = self.ea11(self.eb11(self.ec11(x)))
        xe12 = self.ea12(self.eb12(self.ec12(xe11)))
        xp1  = self.pool1(xe12)

        xe21 = self.ea21(self.eb21(self.ec21(xp1)))
        xe22 = self.ea22(self.eb22(self.ec22(xe21)))
        xp2  = self.pool2(xe22)

        xe31 = self.ea31(self.eb31(self.ec31(xp2)))
        xe32 = self.ea32(self.eb32(self.ec32(xe31)))
        xp3  = self.pool3(xe32)

        xe41 = self.ea41(self.eb41(self.ec41(xp3)))
        xe42 = self.ea42(self.eb42(self.ec42(xe41)))
        xp4  = self.pool4(xe42)

        xe51 = self.ea51(self.eb51(self.ec51(xp4)))
        xe52 = self.ea52(self.eb52(self.ec52(xe51)))

        # Decoder
        xu11 = self.upconv1(xe52)
        xu12 = torch.cat([xu11, xe42], dim=1)
        xd11 = self.da11(self.db11(self.dc11(xu12)))
        xd12 = self.da12(self.db12(self.dc12(xd11)))

        xu21 = self.upconv2(xd12)
        xu22 = torch.cat([xu21, xe31], dim=1)
        xd21 = self.da21(self.db21(self.dc21(xu22)))
        xd22 = self.da22(self.db22(self.dc22(xd21)))

        xu31 = self.upconv3(xd22)
        xu32 = torch.cat([xu31, xe22], dim=1)
        xd31 = self.da31(self.db31(self.dc31(xu32)))
        xd32 = self.da32(self.db32(self.dc32(xd31)))

        xu41 = self.upconv4(xd32)
        xu42 = torch.cat([xu41, xe11], dim=1)
        xd41 = self.da41(self.db41(self.dc41(xu42)))
        xd42 = self.da42(self.db42(self.dc42(xd41)))
        
        # Output layer
        out = self.outconv(xd42)

        return out


#####################    Evaluation metrics   ######################################################

def compute_metrics(truth, pred, threshold=0.5):
    epsilon = 1e-4
    pred_thresh  = np.where(pred>threshold,1,0) #thresholding
    
    # flatten vectors
    truth_flatten = truth.flatten()
    pred_thresh_flatten   = pred_thresh.flatten()

    total_pixels = truth.shape[0]*truth.shape[1]
    
    # Accuracy
    acc = accuracy_score(truth_flatten, pred_thresh_flatten)
    
    # Precision
    precision = precision_score(truth_flatten, pred_thresh_flatten, average='binary')
    
    # Recall
    recall = recall_score(truth_flatten, pred_thresh_flatten, average='binary')
    
    # F1 score
    f1_score = (precision*recall)/(precision+recall+epsilon)
    
    # IoU
    iou = jaccard_score(y_true=truth_flatten, y_pred=pred_thresh_flatten, average='binary')
    
    return [pred_thresh,[acc,precision,recall,f1_score,iou]]


#####################    Prepare dataset   ###################################################### 

print('reading list of training images ...')
file1 = open(train_list_path)
train_list = file1.readlines()
train_list = list(map(str.strip,train_list))
file1.close()

print('reading list of validation images ...')
file2 = open(validation_list_path)
validation_list = file2.readlines()
validation_list = list(map(str.strip,validation_list))
file2.close()

print('creating the training and validation datasets ...')
train = buildingdataset(train_list,validation_list,train=True)
test  = buildingdataset(train_list,validation_list,train=False)

print('creating dataloaders ...')
train_dl = DataLoader(train, 
                      batch_size=1,
                      shuffle=False)

test_dl = DataLoader(test,
                     batch_size=1,
                     shuffle=False)

#####################    Prepare model   ######################################################

print('preparing model ...')

model = UNet(n_class=2)
'''
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.001, 
                      momentum=0.9)

total_params = sum(p.numel() for p in model.parameters())

print('The total number of parameters is: {}'.format(total_params))
'''
# # Load model to GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print("Model moved to GPU")
else:
    print("GPU is not available, using CPU instead")

#####################        Load model          ######################################################

model.load_state_dict(torch.load('models/fold_{}/Unetmodel_{}.pth'.format(fold,model_selected)))
model.eval()

#####################    Prediction & Evaluation of training data   ######################################################

print(' training data ...')

training_acc_per_sample       = []
training_precision_per_sample = []
training_recall_per_sample    = []
training_f1_score_per_sample  = []
training_iou_per_sample       = []

train_counter = 0

for inputs, labels in train_dl:
    
    inputs = inputs.to(device)
    y_pred  = model(inputs)
    
    label   = labels.detach().numpy()
    y_pred0 = y_pred.cpu().detach().numpy()
    
    # Select prediction numpy arrays for positive class as a mask 
    y_pred0_posi = y_pred0[0,1,:,:]
    
    # Compute metrics
    metric = compute_metrics(label,y_pred0_posi,eval_thrshold)

    training_acc_per_sample.append(metric[1][0])
    training_precision_per_sample.append(metric[1][1])
    training_recall_per_sample.append(metric[1][2])
    training_f1_score_per_sample.append(metric[1][3])
    training_iou_per_sample.append(metric[1][4])

    np.save('prediction/fold_{}/trainset/{}'.format(fold,train_list[train_counter]),metric[0])
    train_counter +=1

# arrange data in  data frame
training_metrics_dict = {}
training_metrics_dict['image']     = train_list
training_metrics_dict['accuracy']  = training_acc_per_sample
training_metrics_dict['precision'] = training_precision_per_sample
training_metrics_dict['recall']    = training_recall_per_sample
training_metrics_dict['f1_score']  = training_f1_score_per_sample
training_metrics_dict['iou']       = training_iou_per_sample

df_training_metrics = pd.DataFrame.from_dict(training_metrics_dict)                                                                                   
df_training_metrics.to_csv('training_evaluation_metrics_{}.csv'.format(fold))


#####################    Evaluation of validation data   ######################################################

print('Validation data ...')

validation_acc_per_sample       = []
validation_precision_per_sample = []
validation_recall_per_sample    = []
validation_f1_score_per_sample  = []
validation_iou_per_sample       = []

val_counter = 0

for inputs, labels in test_dl:
    
    inputs = inputs.to(device)
    y_pred  = model(inputs)
    
    label   = labels.detach().numpy()
    y_pred0 = y_pred.cpu().detach().numpy()
    
    # Select prediction numpy arrays for positive class as a mask 
    y_pred0_posi = y_pred0[0,1,:,:]
    
    
    metric = compute_metrics(label,y_pred0_posi,eval_thrshold)
    
    validation_acc_per_sample.append(metric[1][0])
    validation_precision_per_sample.append(metric[1][1])
    validation_recall_per_sample.append(metric[1][2])
    validation_f1_score_per_sample.append(metric[1][3])
    validation_iou_per_sample.append(metric[1][4])
    
    np.save('prediction/fold_{}/validationset/{}'.format(fold,validation_list[val_counter]),metric[0])
    val_counter +=1
                                                                                                          
# arrange data in  data frame
validation_metrics_dict = {}
validation_metrics_dict['image']     = validation_list
validation_metrics_dict['accuracy']  = validation_acc_per_sample
validation_metrics_dict['precision'] = validation_precision_per_sample
validation_metrics_dict['recall']    = validation_recall_per_sample
validation_metrics_dict['f1_score']  = validation_f1_score_per_sample
validation_metrics_dict['iou']       = validation_iou_per_sample

df_validation_metrics = pd.DataFrame.from_dict(validation_metrics_dict)                                                                                   
df_validation_metrics.to_csv('validation_evaluation_metrics_{}.csv'.format(fold))

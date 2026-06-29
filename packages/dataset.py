import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
import numpy as np
import random
from torchvision.io import read_image 
import copy
class buildingdataset(Dataset):
    # load the dataset
    def __init__(self, data_path,image_folder, label_folder, data = 'train', img_extension='.png',lab_extension='.png', transform=None):
        # image_folder: path to training image folder 
        # label_folder:  path to training label folder
        # image_type: img or tci 
        
        self.data_path     = data_path
        self.image_folder  = image_folder  
        self.label_folder  = label_folder
        self.data          = data
        self.img_extension = img_extension
        self.lab_extension = lab_extension
        self.transform     = transform
        
        
        if self.data == 'train':
            
            self.image_path  = '{}/{}/train'.format(self.data_path,self.image_folder)
            self.label_path  = '{}/{}/train'.format(self.data_path,self.label_folder)
            self.images_list = os.listdir(self.image_path)
            #self.images_list = self.images_list[0:80]
            
        elif self.data == 'validate':
            self.image_path  = '{}/{}/validate'.format(self.data_path,self.image_folder)
            self.label_path  = '{}/{}/validate'.format(self.data_path,self.label_folder)
            self.images_list = os.listdir(self.image_path)
            #self.images_list = self.images_list[0:20]
            
        elif self.data == 'test':
            self.image_path  = '{}/{}/test'.format(self.data_path,self.image_folder)
            self.label_path  = '{}/{}/test'.format(self.data_path,self.label_folder)
            self.images_list = os.listdir(self.image_path)
            #self.images_list = self.images_list[0:20]
        else:
            print("Make sure to choose 'train', 'validate', or 'test' ")
            
        
    # get the total number of images in the dataset
    def __len__(self):
        self.dataset_size = len(self.images_list)
        return self.dataset_size
         
    def __getitem__(self, idx):
        
        self.image_name = self.images_list[idx]
        self.image_name = self.image_name.split('.')[0]
        self.namesplit  = self.image_name.split('_') 
        self.label_name = '{}_{}_{}_{}_{}_{}_{}_lab_{}_{}'.format(self.namesplit[0],
                                                          self.namesplit[1],
                                                          self.namesplit[2],
                                                          self.namesplit[3],
                                                          self.namesplit[4],
                                                          self.namesplit[5],
                                                          self.namesplit[6],
                                                          self.namesplit[8],
                                                          self.namesplit[9])
        
        if self.img_extension == '.npy':
            img = np.load('{}/{}.npy'.format(self.image_path, self.image_name))
            img = img.float()
            
            
            if img.shape[2] == 3:
                img = np.moveaxis(img,2,0)      
            
            img = torch.div(img,255)
            
        elif self.img_extension == '.png':
            img = read_image('{}/{}.png'.format(self.image_path, self.image_name))
            img = img.float()
            
        else:
            print('please provide images in "npy" or "png" format')
            
            
        if self.lab_extension == '.npy':
            lab = np.load('{}/{}.npy'.format(self.label_path,self.label_name))
            
        elif self.lab_extension == '.png':
            lab = read_image('{}/{}.png'.format(self.label_path,self.label_name))
            
        else:
            print('please provide labels in "npy" or "png" format')

        
        if self.transform is not None:

            img_np = img.numpy()
            img_np = np.moveaxis(img_np,0,-1)
            
            lab_np = copy.copy(lab)
            
            transformed = self.transform(image=img_np, mask=lab_np)
            img_aug = transformed['image']
            lab_aug = transformed['mask']

            img = np.moveaxis(img_aug,-1,0)
            img = torch.from_numpy(img)
            img = torch.div(img,255)
            lab = torch.from_numpy(lab_aug)
            lab = lab.unsqueeze(0)
            lab = lab.type(torch.LongTensor)
            return [img,lab]
        
        else:
            img = torch.div(img,255)
        
            lab = torch.from_numpy(lab)
            lab = lab.unsqueeze(0)
            lab = lab.type(torch.LongTensor)
            
            return [img,lab]
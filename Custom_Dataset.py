##IMPORT 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class KVasir_dataset(Dataset):
    def __init__(self,train_path,mask_path,transforms=None): #
        super().__init__()
        self.train_path      = train_path
        self.mask_path       = mask_path
        self.tr    = transforms

    def __len__(self):
         return len(self.train_path)
    
    def __getitem__(self,index):        

        #if 'jpg' in self.image_dir_list:
            image = Image.open(self.train_path[index])
            image = np.array(image,dtype=float)
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            image = self.tr(image)
            image = image/255.0 
            #if self.transforms is not None:
             #   image = self.transforms(image)
        
            #if 'jpg' in self.image_dir_list:
            mask = Image.open(self.mask_path[index]).convert('L')
            mask = np.array(mask,dtype=float)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            mask = mask[None,:]
            #mask = np.transpose(mask, (2, 0, 1))
            mask = self.tr(mask)
            mask = mask/255.0 


            return image, mask



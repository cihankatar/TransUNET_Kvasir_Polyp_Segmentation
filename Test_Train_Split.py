import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import random_split
import cv2

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split(all_list, all_path, data_path):
    for list, path, data_path in zip(all_list, all_path, data_path):
        os.chdir(path)
        for idx in range(len(list)):
            img_dir = os.path.join(data_path,list[idx])
            img=Image.open(img_dir)
            img=np.array(img,dtype=float)
            img = img[:, :, [2, 1, 0]]
            cv2.imwrite(list[idx], img)     


def data_path(main_path):
    im_path         = main_path + "/data/images"
    mask_path       = main_path + "/data/masks"
    data_path       = [im_path,mask_path,im_path,mask_path]
    return data_path


def dir_list(data_paths):
    images_dir_list = sorted(os.listdir(data_paths[0])) 
    mask_dir_list   = sorted(os.listdir(data_paths[1]))

    split_ratio     = [int(len(images_dir_list)*0.8),int(len(images_dir_list)*0.2)]
    train_idx,test_idx     = random_split(images_dir_list, split_ratio, generator=torch.Generator().manual_seed(42))
    train_masks,test_masks = random_split(mask_dir_list, split_ratio, generator=torch.Generator().manual_seed(42))

    im_train_list    = [images_dir_list[i] for i in train_idx.indices]
    im_test_list     = [images_dir_list[i] for i in test_idx.indices]
    masks_train_list = [mask_dir_list[i] for i in train_masks.indices]
    masks_test_list  = [mask_dir_list[i] for i in test_masks.indices]
    all_list         = [im_train_list,masks_train_list,im_test_list,masks_test_list]
    return all_list

def split_main():

    main_path   = os.getcwd()
    
    im_train_path    = main_path + "/train/images"
    masks_train_path = main_path + "/train/masks"
    im_test_path     = main_path + "/test/images"
    masks_test_path  = main_path + "/test/masks"
    all_path         = [im_train_path,masks_train_path,im_test_path,masks_test_path]

    for path in all_path:
        if not os.path.exists(path):
            create_dir(path)

    data_paths  = data_path(main_path)
    all_list    = dir_list(data_paths)

    split(all_list, all_path, data_paths)
    os.chdir(main_path)

#split_main()
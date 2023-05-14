import os
from torch.utils.data import DataLoader
from torchvision import transforms
from Custom_Dataset import KVasir_dataset
#from Test_Train_Split import split_main
from glob import glob

def loader(batch_size,num_workers,shuffle):

    train_im_path   = "train/images"
    train_mask_path = "train/masks"
    test_im_path    = "test/images"
    test_mask_path  = "test/masks"

  #  if not os.path.exists(train_im_path):
   #     split_main()

    transformations = transforms.Compose([transforms.CenterCrop(512),transforms.Resize(128,antialias=True)])
    
    train_im_path   = sorted(glob("train/images/*"))
    train_mask_path = sorted(glob("train/masks/*"))

    test_im_path    = sorted(glob("test/images/*"))
    test_mask_path  = sorted(glob("test/masks/*"))

    data_train  = KVasir_dataset(train_im_path,train_mask_path,transformations)
    data_test   = KVasir_dataset(test_im_path, test_mask_path, transformations)

    train_loader = DataLoader(
        dataset     = data_train,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers)
    
    test_loader = DataLoader(
        dataset     = data_test,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers)
    
    return train_loader,test_loader

    
#loader()
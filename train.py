
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm, trange
from one_hot_encode import one_hot,label_encode
from data_loader import loader
from Loss import Dice_CE_Loss
import time as timer

from Unet import UNET
from transunet import TransUNet_copy
from transunet_c import TransUNET_c
from transunet_c_wavelet import TranswaveUNET_c

import numpy as np

def main():

    n_classes   = 1
    batch_size  = 2
    num_workers = 2
    epochs      = 100
    l_r         = 0.001

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)
    
    model1 = UNET(n_classes).to(device)             #1.38      4.47
    model2 = TransUNet_copy(img_dim=128,in_channels=3,out_channels=128,head_num=4,mlp_dim=512,block_num=8,encoder_scale=16,class_num=1).to(device) #1.5  5.3
    model3 = TransUNET_c(n_classes).to(device)       #2.22    8.64
    model4 = TranswaveUNET_c(n_classes).to(device)   #2.27    3.07

    all_models=[model1,model2,model3,model4]


    for idx,model in enumerate(all_models):
        
        best_valid_loss = float("inf")
        print(f"TRAINING FOR MODEL{idx+1} = {model.__class__.__name__}")
        checkpoint_path = "modelsave/checkpoint_model"+str(idx+1)

        optimizer = Adam(model.parameters(), lr=l_r)
        loss_function = Dice_CE_Loss()


        #if torch.cuda.is_available():
        #    model.load_state_dict(torch.load(checkpoint_path))
        #else: 
        #    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

        for epoch in trange(epochs, desc="Training"):

            epoch_loss = 0.0
            model.train()

            #for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            for batch in train_loader:

                images,labels   = batch  
                images,labels   = images.to(device), labels.to(device) 

                start=timer.time()

                model_output    = model(images)
                
                if n_classes == 1:
                    
                    model_output     = model_output.squeeze()
                    #label           = label_encode(labels)
                    train_loss       = loss_function.Dice_BCE_Loss(model_output, labels)

                else:
                    model_output    = torch.transpose(model_output,1,3) 
                    targets_m       = one_hot(labels,n_classes)
                    loss_m          = loss_function.CE_loss_manuel(model_output, targets_m)

                    targets_f       = label_encode(labels) 
                    train_loss      = loss_function.CE_loss(model_output, targets_f)


                epoch_loss     += train_loss.item() 
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                end=timer.time()
                #print(f"batch loss = {train_loss}")

            epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Epoch loss for Model{idx+1} = {model.__class__.__name__} : {epoch_loss}")

            valid_loss = 0.0
            model.eval()

            with torch.no_grad():
                #for batch in tqdm(test_loader, desc=f" Epoch {epoch + 1} in validation", leave=False):
                
                for batch in (test_loader):
                    images,labels   = batch  
                    images,labels   = images.to(device), labels.to(device)   
                    model_output    = model(images)
                    loss            = loss_function.Dice_BCE_Loss(model_output, labels)
                    valid_loss     += loss.item()
                    
                valid_epoch_loss = valid_loss/len(test_loader)

            if valid_epoch_loss < best_valid_loss:

                print(f"previous val loss: {best_valid_loss:2.4f} new val loss: {valid_epoch_loss:2.4f}. Saving checkpoint: {checkpoint_path}")
                best_valid_loss = valid_epoch_loss
                torch.save(model.state_dict(), checkpoint_path)
            
            print(f'\n Model{idx+1} = {model.__class__.__name__} = training Loss: {epoch_loss:.3f}, val. Loss: {valid_epoch_loss:.3f}')


if __name__ == "__main__":
   main()

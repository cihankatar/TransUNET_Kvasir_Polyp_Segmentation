
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm, trange
from one_hot_encode import one_hot,label_encode
from data_loader import loader
from Model import UNET
from Loss import Dice_CE_Loss

def main():

    checkpoint_path = "modelsave/checkpoint.pth"
    
    best_valid_loss = float("inf")

    n_classes   = 1
    batch_size  = 2
    num_workers = 2
    epochs      = 30
    l_r         = 0.001


    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)
    model     = UNET(n_classes)
    #model.load_state_dict(torch.load(checkpoint_path))

    optimizer = Adam(model.parameters(), lr=l_r)
    loss_function      = Dice_CE_Loss()
    epoch_loss = []
    idx=0
    for epoch in trange(epochs, desc="Training"):

        train_loss = 0.0
        loss_sum = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):

            images,labels   = batch                
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


            train_loss     += train_loss.detach().item() / len(train_loader)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            loss_sum = train_loss + loss_sum

           # print(f"  training loss for one batch: {train_loss} ")
        epoch_loss.append(loss_sum.detach().numpy()/len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Epoch loss = {epoch_loss[idx]}")

       #if epoch>2:

        valid_loss = 0.0
        valid_epoch_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f" Epoch {epoch + 1} in validation", leave=False):

                images,labels = batch
                model_output  = model(images)
                loss          = loss_function.Dice_BCE_Loss(model_output, labels)
                valid_loss   += loss.item()

            valid_epoch_loss = valid_loss/len(test_loader)

        if valid_epoch_loss < best_valid_loss:

            print(f"previous val loss: {best_valid_loss:2.4f} new val loss: {valid_epoch_loss:2.4f}. Saving checkpoint: {checkpoint_path}")
            best_valid_loss = valid_epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
        
        print(f'\n Training Loss: {epoch_loss[idx]:.3f} Val. Loss: {valid_epoch_loss:.3f}')
        idx+=1

'''     
        DEBUG --> Copy and paste below lines in order to see current output --<  DEBUG

        
def pred():
    sigmoid_f  = nn.Sigmoid()
    im_test    = np.array(images[0]*255,dtype=int)
    im_test    = np.transpose(im_test, (2,1,0))
    label_test = np.array(labels[0],dtype=int)
    label_test = np.transpose(label_test)
    prediction = sigmoid_f(model_output[0])

    if n_classes>1:
        prediction = torch.argmax(prediction,dim=2)    #for multiclass_segmentation

    else:
        prediction   = prediction.squeeze()
        
    prediction   = prediction.detach().numpy()
    prediction   = prediction > 0.5
    prediction   = np.array(prediction, dtype=np.uint8)
    prediction   = np.transpose(prediction)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im_test)
    plt.subplot(1, 3, 2)
    plt.imshow(label_test,cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction,cmap='gray')

'''

if __name__ == "__main__":
   main()
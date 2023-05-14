from operator import add
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
from data_loader import loader

from Unet import UNET
from transunet import TransUNet_copy
from transunet_c import TransUNET_c
from transunet_c_wavelet import TranswaveUNET_c

def calculate_metrics(y_true, y_pred):

    y_true = y_true.numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard   = jaccard_score(y_true, y_pred)
    score_f1        = f1_score(y_true, y_pred)
    score_recall    = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc       = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


if __name__ == "__main__":

    test_x = sorted(glob("test/images/*"))
    test_y = sorted(glob("test/masks/*"))

    checkpoint_path = "modelsave/checkpoint.pth"
    n_classes   = 1
    n_classes   = 1
    batch_size  = 2
    num_workers = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = TransUNet_copy(img_dim=128,
    #                      in_channels=3,
    #                      out_channels=128,
    #                      head_num=4,
    #                      mlp_dim=512,
    #                      block_num=8,
    #                      encoder_scale=16,
    #                     class_num=1).to(device)
    
       
    model = TransUNET_c(n_classes).to(device)
    
    #model = TranswaveUNET_c(n_classes).to(device)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
    else: 
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    model.eval()
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    
    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)

    for batch in tqdm(test_loader, desc=f"testing ", leave=False):
        images,labels   = batch                
        model_output    = model(images)

        with torch.no_grad():

            model_output    = model(images)
            prediction = torch.sigmoid(model_output)

            if n_classes>1:
                prediction = torch.argmax(prediction,dim=2)    #for multiclass_segmentation

            else:
                

                score = calculate_metrics(labels, prediction)
                metrics_score = list(map(add, metrics_score, score))


        jaccard     = metrics_score[0]/len(test_loader)
        f1          = metrics_score[1]/len(test_loader)
        recall      = metrics_score[2]/len(test_loader)
        precision   = metrics_score[3]/len(test_loader)
        acc         = metrics_score[4]/len(test_loader)
        print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
        print(len(test_loader))


'''
#### to show current output of model 

prediction   = prediction.squeeze()         
prediction   = prediction.detach().numpy()
prediction    = prediction[0]        ## (1, 512, 512)
prediction    = np.squeeze(prediction)     ## (512, 512)
prediction    = prediction > 0.5
prediction    = np.array(prediction, dtype=np.uint8)

im_test    = np.array(images[0]*255,dtype=int)
im_test    = np.transpose(im_test, (2,1,0))
label_test = np.array(labels[0],dtype=int)
label_test = np.transpose(label_test)

prediction = np.transpose(prediction)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(im_test)
plt.subplot(1, 3, 2)
plt.imshow(label_test,cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(prediction,cmap='gray')
plt.imshow(prediction,cmap='gray')
'''

##IMPORT 

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from patchify import patchify


##PATCHFY FUNCTION & POSITIONAL_EMB

def patching_func(images,patch_size):

    n,c,h_iamge,w_image=images.shape
    n_patch=0
    n_patch_per_img = int((h_iamge/patch_size)**2)
    patch_vector=int(patch_size**2)
    
    images=images.numpy()           # patchify function needs numpy
    images=images.squeeze()

    total_patches = np.zeros((n,n_patch_per_img,patch_vector))
 
    for idx,image in enumerate(images):
        patching = patchify(image, (patch_size,patch_size),step=4) # split image into 7x7 small 4x4 patches.patchify(image, (3, 3), step=1)
        for i in range(patching.shape[0]):
            for j in range (patching.shape[1]):
                single_patch=patching[i,j,:,:]
                total_patches[idx,n_patch,:]=single_patch.flatten()
                n_patch=n_patch+1
                if n_patch==49:
                    n_patch=0
    return torch.tensor(total_patches).float()

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

# CLASSES VIT

class ViT (nn.Module):
    def __init__(self, images_shape, patch_size=4, t_blocks=2, token_dim=8, n_heads=2, output_dim=10, mlp_layer_size=8):
        super().__init__()
        
        self.c,self.h_image,self.w_image=images_shape
        
        self.patch_size = patch_size
        self.t_blocks   = t_blocks
        self.n_heads    = n_heads
        self.token_dim  = token_dim
        self.output_dim = output_dim
        self.mlp_layer_size     =   mlp_layer_size

        self.linear_map         =   nn.Linear(int(patch_size**2),token_dim)
        self.class_token        =   nn.Parameter(torch.rand(1, token_dim))
        self.blocks             =   nn.ModuleList([ViTBlock(token_dim, mlp_layer_size,n_heads) for _ in range(t_blocks)])
        self.linear_classifier  =   nn.Linear(token_dim, output_dim)
        self.output_pr          =   nn.Softmax()
        

    def forward(self, images):

        self.n_images,self.c,self.h_image,self.w_image = images.shape
        all_class_token = self.class_token.repeat(self.n_images, 1, 1) 

        '''
        all_class_token = torch.zeros((self.n_images, 1 , self.token_dim))
        for i in range(self.n_images):
            all_class_token[i,:,:] = self.class_token
        '''
        patches     = patching_func(images, self.patch_size)
        linear_emb  = self.linear_map(patches)
        positional_embeddings = get_positional_embeddings(int((self.h_image/self.patch_size)**2+1),int(self.token_dim))

        tokens      = torch.cat((all_class_token,linear_emb),dim=1)
        out         = tokens    +  positional_embeddings.repeat(self.n_images, 1, 1)    # positional embeddings will be added

        for block in self.blocks:
            out = block(out)
            
        out = out[:, 0, :]
        out = self.output_pr(self.linear_classifier(out))
        return out 


class ViTBlock(nn.Module):

    def __init__(self, token_dim, mlp_layer_size=8, num_heads=2):
        super().__init__() 
        self.token_dim      = token_dim
        self.num_heads      = num_heads
        self.mlp_layer_size = mlp_layer_size

        self.layer_norm1    = nn.LayerNorm(token_dim)
        self.layer_norm2    = nn.LayerNorm(token_dim) 
        self.msa            = MSA_Module(token_dim, num_heads)
        self.act_layer      = nn.GELU()
        self.mlp            = nn.Sequential(nn.Linear(token_dim, mlp_layer_size), self.act_layer, nn.Linear(mlp_layer_size, token_dim) )

    def forward(self, x):
        input = self.layer_norm1(x)
        out = x + self.msa(input)
        out = self.layer_norm2(out)
        out = out + self.mlp(out)
        return out
        
class MSA_Module(nn.Module):

    def __init__(self, token_dim, n_heads=2):
        super().__init__() 
        self.n_heads    = n_heads
        self.token_dim  = token_dim

        self.q_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.k_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.v_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.softmax    = nn.Softmax(dim=-1)
        self.linear_map = nn.Linear(150,50)   #  ???????????????
        #self.linear_l = torch.rand(self.n,self.number_tokens,self.number_tokens*self.n_heads)
        self.conv1d = nn.Conv1d(in_channels =100,out_channels=50,kernel_size=1,stride=1)


    def forward (self, tokens):
        
        self.n,self.number_tokens,self.token_size = tokens.shape
        result = torch.zeros(self.n,self.number_tokens*self.n_heads,self.token_size)
        out    = torch.zeros((self.n,self.number_tokens,self.token_size))

        for idx,token in enumerate(tokens):   # 128 batch. each of 50x8, token size : 50x8   --> 50x8            
            concat      = torch.zeros(self.n_heads,self.number_tokens,self.token_size)        
            for head in range(self.n_heads):        # number of heads : 2
                q_linear = self.q_layers[head]      # linear (8x8)  == 50x8 --> 50x8
                k_linear = self.k_layers[head]
                v_linear = self.v_layers[head]

                q  = q_linear(token)
                k  = k_linear(token)
                v  = v_linear(token)

                mat_mul = (torch.matmul(q, k.T)) / ((self.number_tokens-1)**0.5)   # 50x8 x 8x50 = 50x50 
                attention_mask  = self.softmax(mat_mul)
                attention        = attention_mask@v
                concat[head,:,:]  = attention 
            result[idx,:,:]=torch.flatten(input=concat, start_dim=0, end_dim=1)
        
        return self.conv1d(result)

        for idx,i in enumerate(result):
            temp=self.linear_map(result[idx].T)
            out[idx]=temp.T
        return out

def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    model = ViT((1, 28 , 28), patch_size=4, t_blocks=2, token_dim=8, n_heads=2, output_dim=10,mlp_layer_size=8)
    N_EPOCHS = 2
    LR = 0.005
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()

    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch

            y_hat = model(x)

            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y)
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")



if __name__ == '__main__':
    main()
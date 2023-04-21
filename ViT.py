##IMPORT 
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


##PATCHFY FUNCTION & POSITIONAL_EMB

def patching(images,patch_size):
    
    img_patches = rearrange(images, 'b c (patch_x h) (patch_y w) -> b (h w) (patch_x patch_y c)',
                                    patch_x=patch_size, patch_y=patch_size)

    return img_patches

def get_positional_embeddings(sequence_length, d):
    
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

# CLASSES VIT

class ViT_c (nn.Module):
    def __init__(self, images_dim, input_channel, token_dim, n_heads, mlp_layer_size, t_blocks, patch_size, classification,out_dim=10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.images_dim = images_dim
        self.c = input_channel
        self.patch_size = patch_size
        self.t_blocks   = t_blocks
        self.n_heads    = n_heads
        self.token_dim  = token_dim
        self.number_token = (self.images_dim//self.patch_size)**2+1
        self.classification = classification

        self.mlp_layer_size     =   mlp_layer_size

        self.linear_map         =   nn.Linear(self.c*(patch_size**2),token_dim)
        self.class_token        =   nn.Parameter(torch.rand(1, token_dim))
        self.blocks             =   nn.ModuleList([ViTBlock(token_dim, self.number_token, mlp_layer_size, n_heads) for _ in range(t_blocks)])
        self.output_pr          =   nn.Softmax()
        
        if self.classification:
            self.linear_classifier = nn.Sequential(nn.Linear(self.token_dim, out_dim), nn.Softmax(dim=-1)  )

    def forward(self, images):

        

        self.n_images,self.c,self.h_image,self.w_image = images.shape
        all_class_token = self.class_token.repeat(self.n_images, 1, 1).to(self.device)

        patches = patching(images, self.patch_size).to(self.device)

        linear_emb  = self.linear_map(patches)
        positional_embeddings = get_positional_embeddings(self.number_token,self.token_dim).to(self.device)

        tokens      = torch.cat((all_class_token,linear_emb),dim=1)
        out         = tokens    +  positional_embeddings.repeat(self.n_images, 1, 1)    # positional embeddings will be added

        for block in self.blocks:
            out = block(out)

        if self.classification:
            self.linear_classifier(out[:, 0, :]) 
        else: out=out[:, 1:, :]               

        return out 


class ViTBlock(nn.Module):

    def __init__(self, token_dim, n_tokens, mlp_layer_size, num_heads):
        super().__init__() 

        self.token_dim      = token_dim
        self.num_heads      = num_heads
        self.mlp_layer_size = mlp_layer_size
        self.n_tokens     = n_tokens

        self.layer_norm1    = nn.LayerNorm(token_dim)
        self.layer_norm2    = nn.LayerNorm(token_dim) 
        self.msa            = MSA_Module(token_dim, n_tokens, num_heads)
        self.act_layer      = nn.GELU()
        self.mlp            = nn.Sequential(nn.Linear(token_dim, mlp_layer_size), self.act_layer, nn.Linear(mlp_layer_size, token_dim))

    def forward(self, x):
        input = self.layer_norm1(x)
        out = x + self.msa(input)
        out = self.layer_norm2(out)
        out = out + self.mlp(out)
        return out
        
class MSA_Module(nn.Module):

    def __init__(self, token_dim, n_tokens, n_heads):
        super().__init__() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_heads    = n_heads
        self.token_dim  = token_dim
        self.n_tokens   = n_tokens

        self.q_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.k_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.v_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
        self.softmax    = nn.Softmax(dim=-1)
        self.linear_map = nn.Linear(n_heads * n_tokens,n_tokens)   

    def forward (self, tokens):
        
        self.n, self.number_tokens, self.token_size = tokens.shape
        result = torch.zeros(self.n, self.number_tokens*self.n_heads, self.token_size).to(self.device)

        for idx,token in enumerate(tokens):   # 128 batch. each of 65x16*16*3, token size : 50x8   --> 50x8            
            concat      = torch.zeros(self.n_heads, self.number_tokens, self.token_size)        
            for head in range(self.n_heads):        # number of heads : 4
                q_linear = self.q_layers[head]      # linear (512x512)  
                k_linear = self.k_layers[head]
                v_linear = self.v_layers[head]

                q  = q_linear(token)   # 65x512 
                k  = k_linear(token)   # 65x512 
                v  = v_linear(token)   # 65x512 

                mat_mul = (torch.matmul(q, k.T)) / ((self.number_tokens-1)**0.5)   # 65x65
                attention_mask  = self.softmax(mat_mul)   # 65x65
                attention        = attention_mask@v       # 65*65 x 65*512   --> 65x512
                concat[head,:,:]  = attention             # 4x65x512
            result[idx,:,:]=torch.flatten(input=concat, start_dim=0, end_dim=1)
        result=torch.transpose(result,1,2) 
        result=self.linear_map(result)

        return torch.transpose(result,1,2)
'''
def main():
    # Loading data

    model = ViT_c(images_dim=128,input_channel=3, token_dim=512,  n_heads=4, mlp_layer_size=1024, t_blocks=2, patch_size=16,classification=False)
    
    print(model(torch.rand(1, 3, 128, 128)).shape)

if __name__ == '__main__':
    main()

'''
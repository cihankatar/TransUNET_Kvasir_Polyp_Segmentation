import torch
import torch.nn as nn
import time
from einops import rearrange
from ViT import ViT_c

###   ---   UNET  & VIT    ---
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv_block=nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
        
       
    def forward(self, inputs):
        conv_block_out=self.conv_block(inputs)

        return conv_block_out

class down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        conv_block_out=self.conv(inputs)
        encoder_output = self.pool(conv_block_out)

        return conv_block_out, encoder_output


class up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        conv_t_out = self.up(inputs)
        concat = torch.concat((conv_t_out,skip),dim=1)
        decoder_output = self.conv(concat)
        return decoder_output
    

class TransUNET_c(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        #sigmoid_f      = nn.Sigmoid()
        #softmax_f      = nn.Softmax()

        size_en=[3,64,128,256,512]
        self.encoder_blocks = nn.ModuleList([down(in_f, out_f) for in_f, out_f in zip(size_en,size_en[1:])])

        self.b = conv_block(512, 1024)

        size_dec=[1024,512,256,128,64]
        self.decoder_blocks = nn.ModuleList([up(in_f, out_f) for in_f, out_f in zip(size_dec,size_dec[1:])])

        
        if self.n_classes > 1:

            self.outputs  = nn.Conv2d(64, 2, kernel_size=1, padding=0)
            #self.outputs = softmax_f(self.outputs)
        else:
            self.outputs  = nn.Conv2d(64, 1, kernel_size=1, padding=0)
            #self.outputs  = sigmoid_f(self.outputs)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.VIT = ViT_c(images_dim=8,
                         input_channel=1024, 
                         token_dim=1024, 
                         n_heads=4, 
                         mlp_layer_size=512, 
                         t_blocks=8, 
                         patch_size=1,
                         classification=False).to(device)
        
    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        s1, p1 = self.encoder_blocks[0](inputs)     # 1x  64 x 128x128 ,  64  x 64x64
        s2, p2 = self.encoder_blocks[1](p1)         # 1x 128 x 64x64   ,  128 x 32x32
        s3, p3 = self.encoder_blocks[2](p2)         # 1x 256 x 32x32   ,  256 x 16x16
        s4, p4 = self.encoder_blocks[3](p3)         # 1x 512 x 16x16   ,  512 x 8x8

        x = self.b(p4)                              # 1x 1024 x 8x8

        x = self.VIT(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=8, y=8)  # 1x 1024 x 8 x 8

        d1 = self.decoder_blocks[0](x, s4)          # 1x 512 x  16x16
        d2 = self.decoder_blocks[1](d1, s3)         # 1x 256 x  32x32
        d3 = self.decoder_blocks[2](d2, s2)         # 1x 128 x  64x64
        d4 = self.decoder_blocks[3](d3, s1)         # 1x  64 x 128x128

        outputs = self.outputs(d4)                  # 1   64 x 128x128

        return outputs

if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 128, 128))
    f = TransUNET_c(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')



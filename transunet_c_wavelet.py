import torch
import torch.nn as nn
import time
from einops import rearrange
from ViT import ViT_c
import pywt


###   ---   UNET  & VIT & WAVELET   ---

def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

    
class down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
       

        self.dwt_conv   = nn.Conv2d(in_c,int(out_c/4),kernel_size=3,padding=1)

        self.conv_block = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Conv2d(out_c, out_c, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
        
       
        self.device=device_f()

    def forward(self, inputs):

        conv_out       = self.conv_block(inputs)

        dwt_conv_out   = self.dwt_conv(inputs)

        dwt_conv_out   = dwt_conv_out.detach().cpu()
        coef           = pywt.dwt2(dwt_conv_out,'db1')
        cA, (cH,cV,cD) = coef
        
        cA = torch.tensor(cA,requires_grad=True).to(self.device)
        cH = torch.tensor(cH,requires_grad=True).to(self.device)
        cV = torch.tensor(cV,requires_grad=True).to(self.device)
        cD = torch.tensor(cD,requires_grad=True).to(self.device)
        
        dwt = torch.concat((cA,cH,cV,cD),dim=1)

        return  dwt,conv_out


class up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU(inplace=True))
        
        self.dwt_up = nn.Sequential(nn.Conv2d(in_c, out_c*2, kernel_size=3, stride=1,padding=1),
                                nn.BatchNorm2d(out_c*2),
                                nn.ReLU(inplace=True))
        
        self.last_up = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1,padding=1),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True))

        self.device  = device_f()

    def forward(self, x, coef=None, s=None):
        
        if coef is not None:
            coef_hp = self.dwt_up(coef).cpu().detach()
            coef = x.cpu().detach(),(coef_hp,coef_hp,coef_hp)
        else:
            coef = x.cpu().detach(),(None,None,None)


        idwt_out = pywt.idwt2(coef, 'db1')
        idwt_out = torch.tensor(idwt_out,requires_grad=True).to(self.device)

        if s is not None:
            concat = torch.concat((idwt_out,s),dim=1)
            decoder_output = self.conv(concat)

        else:
            decoder_output = self.last_up(idwt_out)

        return decoder_output
    

class TranswaveUNET_c(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        
        self.n_classes = n_classes
        self.device = device_f()
        
        self.first_conv = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True))

        size_en=[128,256,512,1024]
        self.encoder_blocks = nn.ModuleList([down(in_f, out_f) for in_f, out_f in zip(size_en,size_en[1:])])


        self.VIT = ViT_c(images_dim=8,
                         input_channel=1024, 
                         token_dim=1024, 
                         n_heads=4, 
                         mlp_layer_size=512, 
                         t_blocks=8, 
                         patch_size=1,
                         classification=False).to(self.device)
        
        self.bottelneck = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True))
        
        size_dec=[1024,256,512,128,256,64,64,16]
        self.decoder_blocks = nn.ModuleList([up(size_dec[0], size_dec[1]), 
                                             up(size_dec[2], size_dec[3]), 
                                             up(size_dec[4], size_dec[5]),
                                             up(size_dec[6], size_dec[7])])

        if self.n_classes > 1:

            self.outputs  = nn.Conv2d(16, 2, kernel_size=1, padding=0)
            #self.outputs = softmax_f(self.outputs)
        else:
            self.outputs  = nn.Conv2d(16, 1, kernel_size=1, padding=0)
            #self.outputs  = sigmoid_f(self.outputs)

        
    def forward(self, inputs):                           # 2x 3  x 128x128
        
        s0 = self.first_conv(inputs)                     # 2x 128  x 64x64

        coef1, s1 = self.encoder_blocks[0](s0)            # 2x 256 x 32x32
        coef2, s2 = self.encoder_blocks[1](coef1)          # 2x 512 x 16x16
        coef3, _  = self.encoder_blocks[2](coef2)          # 2x 1024 x 8x8
        
    
        x = self.VIT(coef3)
        x = rearrange(x, "b (x y) c -> b c x y", x=8, y=8)  # 2x 1024 x 8 x 8

        x = self.bottelneck(x)                              # 2x 512 x 8 x 8

        idwt1 = self.decoder_blocks[0](x,     coef3, s2)        # 2x 256 x  16x16
        idwt2 = self.decoder_blocks[1](idwt1, coef2, s1)        # 2x 128 x  32x32
        idwt3 = self.decoder_blocks[2](idwt2, coef1, s0)        # 2x 64  x  64x64
        d4    = self.decoder_blocks[3](idwt3)                   # 2x 16  x  128x128

        outputs = self.outputs(d4)                             # 2x 1   x 128x128
        
        return outputs

if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 128, 128))

    f = TranswaveUNET_c(1)
    y = f(x)
    print(x.shape)
    print(y.shape)

    end=time.time()
    
    print(f'spending time :  {end-start}')



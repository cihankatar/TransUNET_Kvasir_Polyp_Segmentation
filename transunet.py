import torch
import torch.nn as nn
from einops import rearrange
from ViT import ViT_c
import time

#### ---- (TRANSUNET) CONV & VIT & UPSAMPLE 


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride,):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)       # 256x32x32
        )
        self.convlayer=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) ,
            nn.BatchNorm2d(out_channels),     
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))            # 256x32x32


    def forward(self, x):            # 1x 128 x 64 x 64
        x_down = self.downsample(x)  # 1x 256 x 32 x 32

        x = self.convlayer(x)
        x = x + x_down

        return x
    
class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, encoder_scale):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // encoder_scale

        self.VIT = ViT_c(self.vit_img_dim,
                         out_channels*8, 
                         out_channels*8, 
                         head_num, 
                         mlp_dim, 
                         block_num, 
                         patch_size=1,
                         classification=False).to(device)
        
        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):       # 2x  3 x 128 x 128
        x = self.conv1(x)       # 2x 128 x 64 x 64
        x = self.norm1(x)       
        x1 = self.relu(x)       # 2x 128 x 64 x 64

        x2 = self.encoder1(x1)  # 2x 256 x 32 x 32
        x3 = self.encoder2(x2)  # 2x 512 x 16 x 16
        x = self.encoder3(x3)   # 2x 1024 x 8 x 8

        x = self.VIT(x)

        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # 1x 1024 x 8 x 8

        x = self.conv2(x)       # 1x 512 x 8 x 8
        x = self.norm2(x)
        x = self.relu(x)

        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):  # 1 x 512 x 8x8
        x = self.decoder1(x, x3)       # 1 x 256 x 16x16
        x = self.decoder2(x, x2)       # 1 x 128 x 32x32
        x = self.decoder3(x, x1)       # 1 x 64  x 64x64
        x = self.decoder4(x)           # 1 x 16  x 128x128
        x = self.conv1(x)              # 1 x 1   x 128x128

        return x


class TransUNet_copy(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, encoder_scale, class_num):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, encoder_scale)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

        return x

if __name__ == "__main__":

    start=time.time()

    x = torch.randn((2, 3, 128,128))
    model = TransUNet_copy(img_dim=128,in_channels=3,out_channels=128,head_num=4,mlp_dim=512,block_num=8,encoder_scale=16,class_num=1)
    y = model(x)
    print(x.shape)
    print(y.shape)
    end=time.time()
    print(f'spending time :  {end-start}')

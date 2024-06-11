import torch
from torch import nn

    
class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__() 
        self.depth_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_channels)
        
        self.point_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1)
        
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class GLAE(nn.Module):
    def __init__(self, out_channels=384, out_size=56):
        super().__init__() 
        self.gfc_block = nn.ModuleDict({
                        'd_conv1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=2, padding=1, dilation=2),
                        'relu1': nn.ReLU(inplace=True),
                        'avg_pool2d1': nn.AvgPool2d(kernel_size=2, stride=2)
                        })

        self.gfe_block1 = nn.ModuleDict({
                        'g_conv1': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 59)),
                        'g_conv2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(59, 1)),
                        })
        
        self.gfe_block2 = nn.ModuleDict({
                        'g_conv1': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 29)),
                        'g_conv2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(29, 1)),
                        })
        
        self.intermediate_block = nn.ModuleDict({
                        'relue1': nn.ReLU(inplace=True),
                        'conv1': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=1),
                        'relue2': nn.ReLU(inplace=True),
                        })

        self.lfc_block = nn.ModuleDict({
                'relue1': nn.ReLU(inplace=True),
                'conv1': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                'relue2': nn.ReLU(inplace=True),
                'conv2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=1),
                })

                
        self.decode = nn.Sequential(
                        # decode
                        nn.Upsample(size=8, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=15, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=32, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=63, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=127, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=out_size, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                                stride=1, padding=1)
        )


    def forward(self, x):
        for _, layer in self.gfc_block.items():
            x = layer(x)

        x1 = self.gfe_block1['g_conv1'](x)
        x2 = self.gfe_block1['g_conv2'](x)
        z = torch.matmul(x1, x2)
        
        for _, layer in self.intermediate_block.items():
            z = layer(z)
            
        x1 = self.gfe_block2['g_conv1'](z)
        x2 = self.gfe_block2['g_conv2'](z)
        z = torch.matmul(x1, x2)
        
        for _, layer in self.lfc_block.items():
            z = layer(z)
        z = self.decode(z)
        return z
        

class Student(nn.Module):
    def __init__(self, out_channels=384, padding=False):
        super().__init__() 
        self.pad_mult = 1 if padding else 0
        self.pdn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3 * self.pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * self.pad_mult),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=3 * self.pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * self.pad_mult),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1 * self.pad_mult),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1))
        
    def forward(self, x):
        x = self.pdn(x)
        return x
    
    
class Teacher(nn.Module):
    def __init__(self, out_channels=384, padding=False):
        super().__init__() 
        self.pad_mult = 1 if padding else 0
        self.pdn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3 * self.pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * self.pad_mult),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=3 * self.pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * self.pad_mult),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1 * self.pad_mult),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1))
        
    def forward(self, x):
        x = self.pdn(x)
        return x
    
    
if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256)
    teacher = Teacher()
    glae = GLAE()
    out1 = glae(input)
    out2 = teacher(input)
    print(out1.shape)
    print(out2.shape)
        
    



    
    







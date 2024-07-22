import json
import os 
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,PROJECT_PATH)
import torch
import torch.nn as nn
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self,in_channels, layers_conv,channels,kernel_size,strides,padding,pool_size,drop_prob,pooling):
        self.in_channels = in_channels
        self.layers_conv = layers_conv
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pool_size = pool_size
        self.drop_prob = drop_prob
        self.pooling = pooling

        self.features = self.create_network()

    def create_network(self):
        modules_conv = []
        in_channels = self.in_channels

        for i_conv in range(self.layers_conv):
            modules_conv.append(nn.Conv2d(in_channels,
                                            self.channels[i_conv],
                                            stride = self.strides[i_conv],
                                            padding = self.padding))
            modules_conv.append(nn.BatchNorm2d(self.channels[i_conv]))
            modules_conv.append(nn.ReLu())
            if self.pooling[i_conv]:
                modules_conv.append(nn.MaxPool2d(self.pool_size[0],self.pool_size[1]))
            modules_conv.append(nn.Dropout(p = self.drop_prob_conv))
            in_channels = self.channels[i_conv]
        modules_conv.append(nn.AdaptiveAvgPool2d(output_size= (1,1)))
        return nn.Sequential(*modules_conv)

    def forward(self,x):
        batch_size,feature_size = x.size(0),self.channels[-1]
        out = self.features(x)
        out = out.view(batch_size,feature_size)

        return out
        

    
class ResNet(nn.Module):
    def __init__(self,model_string = 'pytorch/vision:v0.10.0', 
                norm_mean = [0.485, 0.456, 0.406],
                norm_std =  [0.229, 0.224, 0.225], depth = 'resnet18', ):
        super(ResNet,self).__init__()




        
















model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inpt_img = torch.rand(10,3,512,512)
inpt_norm = norm(inpt_img)
out_feat = model(inpt_norm)

class FeatureEncoder()

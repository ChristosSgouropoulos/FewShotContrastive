import os
import sys
import json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import random
import torch.nn as nn
from typing import List
from torchvision import transforms
import torch.nn.functional as F
from models.dropblock import DropBlock

class MLP(nn.Module):
    def __init__(self,model_config,purpose):
        super(MLP,self).__init__()
        self.in_features = model_config[purpose]['in_features']
        self.out_features = model_config[purpose]['out_features']
        self.layers = model_config[purpose]['layers']
        self.drop_prob = model_config[purpose]['drop_prob']
        self.purpose = purpose
        self.mlp = self._create_network()
        

    def _create_network(self):
        modules = []
        in_features = self.in_features
        for out_feature in self.layers:
            modules.append(nn.Linear(in_features = in_features, out_features = out_feature, bias = False))
            if self.purpose !='projection_head':
                modules.append(nn.BatchNorm1d(num_features = out_feature))
                modules.append(nn.Dropout(p = self.drop_prob))
            in_features = out_feature

        modules.append(nn.Linear(in_features = in_features, out_features = self.out_features))

        return nn.Sequential(*modules)
    
    def forward(self,x):
        return self.mlp(x)
    
class CNN(nn.Module):
    def __init__(self,model_config):
        self.in_channels = model_config["CNN"]['in_channels']
        self.layers_conv = model_config["CNN"]['layers_conv']
        self.channels = model_config["CNN"]['channels']
        self.kernel_size = model_config["CNN"]['kernel_size']
        self.strides = model_config["CNN"]['strides']
        self.padding = model_config["CNN"]['padding']
        self.pool_size = model_config["CNN"]['pool_size']
        self.drop_prob = model_config["CNN"]['drop_prob']
        self.pooling = model_config["CNN"]['pooling']

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
    



# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def Res12(model_config ,**kwargs):
    """Constructs a ResNet-12 model.
    """
    keep_prob = model_config['Res12']['keep_prob']
    avg_pool = model_config['Res12']['avg_pool']
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model
















class SelfAttention(nn.Module):
    def __init__(self, model_config):
        super(SelfAttention, self).__init__()
        self.embed_dim = model_config['self_attention_embed_dim']
        self.num_heads = model_config['self_attention_num_heads']
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        
    def forward(self,x):
        x = x.permute(1, 0, 2)  # (augmentations+original, batch_size, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, augmentations+original, embed_dim)
        
        # Flatten the output to get the final embedding of dimension augmentations+original*D
        batch_size, seq_length, embed_dim = attn_output.shape
        output = attn_output.reshape(batch_size, seq_length * embed_dim)  # (batch_size, (augmentations+original) * D)
        
        return output
    
    
if __name__ == '__main__':
    ### Check  ResNet
    with open("models/model_params.json", "r") as f:
        model_config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inpt_img = torch.rand(10,3,84,84)
    inpt_img = inpt_img.to(device)
    model = Res12(model_config)
    model.to(device)
    out_feat = model(inpt_img)
    print("Resnet",out_feat.size())




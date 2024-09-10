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
    

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=1280, output_dim=2560):
        super(ProjectionHead, self).__init__()
        
        # Single layer: Keep the same dimension for input and output
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Use Layer Normalization instead of Batch Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Pass through the first layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        # Pass through the second layer
        x = self.fc2(x)
        x_norm = F.normalize(x, p=2.0, dim=1, eps=1e-12, out=None)

        return x_norm


class Conv4_64(nn.Module):
    def __init__(self,input_channels=3):
        super(Conv4_64, self).__init__()
        
        # Define the 4 convolutional layers, each with 64 filters
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)  # Conv layer 1
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)              # Conv layer 2
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)              # Conv layer 3
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)              # Conv layer 4
        self.bn4 = nn.BatchNorm2d(64)
        

    def forward(self, x):
        # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 3: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 4: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten the tensor to pass into the fully connected layer
        x = F.adaptive_avg_pool2d(x, (1, 1))  # GAP reduces the spatial dimensions to 1x1
        x = x.view(x.size(0), -1) 
        
        return x

class Conv4_512(nn.Module):
    def __init__(self,input_channels=3):
        super(Conv4_512, self).__init__()
        
        # Define the 4 convolutional layers, each with 64 filters
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)  # Conv layer 1
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)              # Conv layer 2
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)              # Conv layer 3
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 512, kernel_size=3, padding=1)              # Conv layer 4
        self.bn4 = nn.BatchNorm2d(512)
        

    def forward(self, x):
        # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 3: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 4: Conv -> BatchNorm -> ReLU -> MaxPool
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten the tensor to pass into the fully connected layer
        x = F.adaptive_avg_pool2d(x, (1, 1))  # GAP reduces the spatial dimensions to 1x1
        x = x.view(x.size(0), -1) 
        
        return x


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
        self.ffn_dim = model_config['self_attention_ffn_dim']
        self.dropout = model_config['dropout']

        # TransformerEncoderLayer: includes MultiheadAttention, FeedForward, and LayerNorm
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            batch_first=True  # Makes the layer expect input as (batch_size, seq_length, embed_dim)
        )
        
    def forward(self, x):
        # Input x shape: (batch_size, 4, D)
        
        # Pass input through TransformerEncoderLayer
        attn_output = self.encoder_layer(x)  # Output shape: (batch_size, 4, D)
        
        # Channel-wise concatenation: Concatenate along the feature dimension to get (batch_size, 4 * D)
        output = torch.cat([attn_output[:, i, :] for i in range(attn_output.size(1))], dim=-1)
        
        return output

class SelfAttention1(nn.Module):
    def __init__(self, model_config):
        super(SelfAttention, self).__init__()
        self.embed_dim = model_config['self_attention_embed_dim']
        self.num_heads = model_config['self_attention_num_heads']
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        
    def forward(self, x):
        # Input x shape: (batch_size, 4, D)
        
        # Permute for MultiheadAttention: (seq_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (4, batch_size, D)
        
        # Apply self-attention
        attn_output, _ = self.multihead_attn(x, x, x)  # (4, batch_size, D)
        
        # Permute back: (batch_size, 4, D)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, 4, D)
        
        # Channel-wise concatenation: Concatenate along the feature dimension to get (batch_size, 4 * D)
        output = torch.cat([attn_output[:, i, :] for i in range(attn_output.size(1))], dim=-1)
        
        return output
    
    
if __name__ == '__main__':
    ### Check  ResNet
    with open("models/model_params.json", "r") as f:
        model_config = json.load(f)

    gpu_index = 1
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    inpt_img1 = torch.rand(10,3,84,84)
    inpt_img1 = inpt_img1.to(device)
    inpt_img2 = torch.rand(10,3,84,84)
    inpt_img2 = inpt_img2.to(device)
    inpt_img3 = torch.rand(10,3,84,84)
    inpt_img3 = inpt_img3.to(device)
    inpt_img4 = torch.rand(10,3,84,84)
    inpt_img4 = inpt_img4.to(device)

    print(inpt_img1)
    model = Res12(model_config).to(device)
    print(next(model.parameters()).device)
    out_feat1 = model(inpt_img1)
    out_feat2 = model(inpt_img2)
    out_feat3 = model(inpt_img3)
    out_feat4 = model(inpt_img4)



    print("Resnet",out_feat1.size())

    modela = Conv4_512(input_channels = 3).to(device)
    out_feata = modela(inpt_img1)
    print("Conv4_64",out_feata.size())

    out_feats = torch.stack([out_feat1,out_feat2,out_feat3,out_feat4], dim =1)
    print(out_feats.size()) 
    attention = SelfAttention(model_config = model_config).to(device)
    after_attention = attention(out_feats)
    print(after_attention.size())
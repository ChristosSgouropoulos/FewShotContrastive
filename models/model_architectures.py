import os
import sys
import json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import random
import torch.nn as nn
from typing import List
from torchvision import transforms

class MLP(nn.Module):
    def __init__(self,model_config,purpose):
        super(MLP,self).__init__()
        self.in_features = model_config[purpose]['in_features']
        self.out_features = model_config[purpose]['out_features']
        self.layers = model_config[purpose]['layers']
        self.drop_prob = model_config[purpose]['drop_prob']
        self.mlp = self._create_network()

    def _create_network(self):
        modules = []
        in_features = self.in_features
        for out_feature in self.layers:
            modules.append(nn.Linear(in_features = in_features, out_features = out_feature, bias = False))
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


class ResNet(nn.Module):
    def __init__(self,model_config):
        super(ResNet,self).__init__()
        self.model_config = model_config
        print(model_config)
        self.model_string = model_config['ResNet']['model_string']
        self.depth = model_config['ResNet']['depth']
        self.pretrained = model_config['ResNet']['pretrained']
        self.norm_mean = model_config['ResNet']['norm_mean']
        self.norm_std = model_config['ResNet']['norm_std']
        self.model = torch.hub.load(self.model_string, self.depth, self.pretrained)

    def normalize_image(self,img_tensor):
        normalizer = transforms.Normalize(mean=self.norm_mean, std = self.norm_std)
        normalized_img = normalizer(img_tensor)
        return normalized_img
    
    def forward(self,x):
        x_normalized = self.normalize_image(x)
        x_out = self.model(x_normalized)

        return x_out
    
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
    inpt_img = torch.rand(1,3,256,256)
    model = ResNet(model_config = model_config)
    out_feat = model(inpt_img)
    print("Resnet",out_feat.size())


    ### Check Self Attention
    x1 = torch.rand(1,1000)
    x2 = torch.rand(1,1000)
    x3 = torch.rand(1,1000)
    x4 = torch.rand(1,1000)
    feature_list = [x1,x2,x3,x4]

    concatenator = SelfAttention(model_config = model_config, shuffle = False)
    feature = concatenator(feature_list)
    print("Self Attention",feature.size())



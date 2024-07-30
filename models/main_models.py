import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import random
import torch.nn as nn
import json
from models.model_architectures import CNN,MLP,SelfAttention,ResNet
from utils.augmentations import image_augmentations,audio_augmentations


class EncoderModule(nn.Module):
    def __init__(self,config_file,augmentation_module):
        super(EncoderModule, self).__init__()
        self.augmentation_module = augmentation_module
        self.config_file = config_file
        self.encoder, self.model_config = get_encoder(encoder_name = self.config_file['encoder'])

    def forward(self,x):
        ## Get a fixed number of augmentations of x in x_list
        x_list = self.augmentation_module(x)
        ## get_encoder
        encoded_features = []
        for x in x_list:
            encoded_x = self.encoder(x)
            ## Encoded x will be of shape [batch_size,D]
            encoded_features.append(encoded_x)
        return encoded_features
    
    def _get_model_config(self):
        return self.model_config


def get_encoder(encoder_name):
    with open("models/model_params.json", "r") as f:
        model_config = json.load(f)
    if encoder_name == 'CNN':
        encoder = CNN(model_config = model_config)
    elif encoder_name == 'ResNet':
        encoder = ResNet(model_config = model_config)
    return encoder,model_config



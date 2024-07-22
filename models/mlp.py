import os
import sys
import json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from typing import List
class MLP(nn.Module):
    def __init__(self,in_features: int, layers: List[int], drop_prob: float, out_features: int):
        super(MLP,self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.drop_prob = drop_prob

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
        return self.classifier(x)
    
    
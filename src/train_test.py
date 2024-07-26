import sys
import os 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.task_sampler import TaskSampler
from torch.utils.data import DataLoader
from datasets.miniimagenet import MiniImageNet
from pathlib import Path
import random
from statistics import mean
import numpy as np
import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
from datasets.task_sampler import TaskSampler
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR


mini_imagenet_path = "data/mini_imagenet/images"
train_set = MiniImageNet(root=mini_imagenet_path, split="train", training=True)
train_sampler = TaskSampler(train_set, n_way=5, n_shot=5, n_query=10, n_tasks=100)
train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
for batch in train_loader:
    print(batch[3])
    break
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.task_sampler import TaskSampler
from torch.utils.data import DataLoader
import tqdm
import torch

def training_loop(data_loader, optimizer,encoder_module):
    all_loss = []
    with tqdm(enumerate(data_loader), total = len(data_loader), desc= "Training") as tqdm_train:
        for episode_index,support_input,support_labels,query_input,query_labels,_ in tqdm_train:
            support_features = encoder_module(support_input)
            ## Compute prototypes
            unique_labels = support_labels.unique()
            prototype_list = []
            for label in unique_labels:
                # Get indices of features belonging to the current label
                label_indices = (support_labels == label).nonzero(as_tuple=True)[0]
                label_features = support_features[label_indices]
                mean_feature = label_features.mean(dim=0)
                prototype_list.append({"prototype":mean_feature,"label":label})

            ## Now do the same with the query set
            query_features = encoder_module(query_input) ## Query features of size : [batch_size,(original+augmentations)*D]
            label_indices = (support_labels == label).nonzero(as_tuple=True)[0]
            label_features = support_features[label_indices]


            # Convert the prototype list to a tensor (optional)
            prototypes= torch.stack(prototype_list)

            






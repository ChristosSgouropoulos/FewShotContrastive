import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.task_sampler import TaskSampler
from torch.utils.data import DataLoader
from models.model_architectures import SelfAttention
from models.main_models import EncoderModule, get_encoder
from utils.augmentations import image_augmentations
from loops.loss import FSL_loss,cos_similarity
import random
import tqdm
import torch
import json 

def training_loop(data_loader, optimizer,experiment_config, hyperparameter_m, hyperparameter_T):
    with open(experiment_config, "r") as f:
        experiment_config = json.load(f)    
    ## Get the encoder
    encoder = EncoderModule(config_file = experiment_config,augmentation_module = image_augmentations)
    ## Get the model_config dictionary to use for attention
    model_config = encoder._get_model_config()
    ## Initialize two attention layers, one with suffle and one without

    attention_layer = SelfAttention(model_config = model_config, shuffle = False)
    shuffled_attention_layer = SelfAttention(model_config= model_config, shuffle = True)
    
    ## Iterate data loader 
    with tqdm(enumerate(data_loader), total = len(data_loader), desc= "Training") as tqdm_train:
        for episode_index,support_input,support_labels,query_input,query_labels,_ in tqdm_train:
            support_feature_list = encoder(support_input)
            support_features = torch.stack(support_feature_list, dim=1)
            support_features = attention_layer(support_features) ### [batch_size, (len(augmentations)+1)*D]
            ## Compute prototypes
            ## Unique sorts labels 
            unique_labels = support_labels.unique()
            prototype_list = []
            for label in unique_labels:
                # Get indices of features belonging to the current label
                label_indices = (support_labels == label).nonzero(as_tuple=True)[0]
                label_features = support_features[label_indices]
                prototype = label_features.mean(dim=0)
                prototype_list.append(prototype)
            stacked_prototypes = torch.stack(prototype_list, dim = 0) ## [num_of_classses,D]
            ## Now forward pass query set
            query_feature_list = encoder(query_input)
            query_features = torch.stack(query_feature_list,dim = 1)
            query_features = attention_layer(query_features)
            ## Okay now need to sort query features by the labels
            sorted_query_labels, indices = torch.sort(query_labels)
            sorted_query_features = query_features[indices]
            ## Compute FSL_loss:
            FSL_loss = FSL_loss(prototypes = stacked_prototypes, query_features = sorted_query_features,query_labels = sorted_query_labels)
            ## Now lets start making the contrastive part
            ## Query features will again pass through the feature extractor get shuffled and pass through the attention layer again:
            augmentations = query_feature_list[1:]
            random.shuffle(augmentations)
            shuffled_query_features = torch.stack([query_feature_list[0]] + augmentations, dim=1)  # (batch_size, augmentations+original, embed_dim)
            query_features_contrastive = attention_layer(shuffled_query_features)
            sorted_query_features_contrastive = query_features_contrastive[indices]
            ### Now costruct the positive examples
            













            






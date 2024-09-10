import os 
import sys 
import json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from models.main_models import EncoderModule
from models.model_architectures import SelfAttention
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score 
def prototypical_inference1(sorted_query_features,prototypes):
    predictions = []
    for feature in sorted_query_features:
        distance_list = []
        for prototype in prototypes:
            eucl_distance = torch.cdist(prototype.unsqueeze(0), feature.unsqueeze(0), p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
            distance_list.append(eucl_distance)
        ## Get the label of minimum distance:
        min_value = min(distance_list)  
        predicted_label = distance_list.index(min_value)  
        
        predictions.append(predicted_label)
    return predictions


def prototypical_inference(sorted_query_features,prototypes):
    predictions = []
    for feature in sorted_query_features:
        distance_list = []
        for prototype in prototypes:
            eucl_distance = torch.cdist(prototype.unsqueeze(0), feature.unsqueeze(0), p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
            distance_list.append(eucl_distance)
        ## Get the label of minimum distance:
        distances = torch.tensor(distance_list)
        probabilities = torch.softmax(-distances,dim = 0)
        predicted_label = torch.argmax(probabilities).item()
        predictions.append(predicted_label)
    return predictions



def testing_loop(test_loader, experiment_config,device):
    with open(experiment_config , 'r') as f:
        experiment_config = json.load(f)
    encoder = torch.load(experiment_config['encoder_pt']).to(device)
    attention = torch.load(experiment_config['attention_pt']).to(device)
    encoder.eval()
    attention.eval()
    accuracy_list = []
    with torch.inference_mode():
            with tqdm(enumerate(test_loader), total = len(test_loader), desc= "Testing") as tqdm_test:
                for few_shot_batch in tqdm_test:
                     episode_index = few_shot_batch[0]
                     support_input,support_labels,query_input,query_labels,_ = few_shot_batch[1]

                     support_input = support_input.to(device)
                     support_labels = support_labels.to(device)
                     query_input = query_input.to(device)
                     query_labels = query_labels.to(device)

                     support_feature_list = encoder(support_input)
                     support_features = torch.stack(support_feature_list, dim=1)
                     support_features = attention(support_features) ### [batch_size, (len(augmentations)+1)*D]
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
                     ## Now forward pass query set
                     query_feature_list = encoder(query_input)
                     query_features = torch.stack(query_feature_list,dim = 1)
                     query_features = attention(query_features)
                     ## Okay now need to sort query features by the labels
                     sorted_query_labels, indices = torch.sort(query_labels)
                     sorted_query_features = query_features[indices]
                     predictions = prototypical_inference(sorted_query_features = sorted_query_features, prototypes = prototype_list)

                     ## Compute Testing accuracy:
                     labels_np = [label.cpu().numpy() for label in sorted_query_labels]
                     episode_accuracy = accuracy_score(predictions,labels_np)
                     accuracy_list.append(episode_accuracy)
    mean_accuracy = np.mean(accuracy_list)
    print(f"Testing Mean Accuracy is : {mean_accuracy}")
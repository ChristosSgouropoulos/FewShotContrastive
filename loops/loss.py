import torch
import torch.nn as nn
def FSL_loss(prototypes, query_features,query_labels):
    """ Prototypes are expected to be a torch tensor of size [classes, D] """
    ## Take a query tensor:
    batch_loss = 0
    for i in range(len(query_features)):
        query_feature = query_features[i]
        query_label = query_labels[i]
        ## The numerator will be the Euclidean distance of the query_feature with the prototype of the same label
        prototype_of_same_label = prototypes[query_label]
        numerator = torch.exp(-torch.cdist(query_feature, prototype_of_same_label, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'))
        ## Now the denominator will be calculated using the sum of the distances of the query feature with all the prototype classes
        denominator = 0
        for j in range(len(prototypes)):
            prototype = prototypes[j]
            to_sum = torch.exp(-torch.cdist(query_feature,prototype, p= 2.0,compute_mode='use_mm_for_euclid_dist_if_necessary'))
            denominator = denominator + to_sum
        instance_loss = -torch.log(numerator/denominator)
        batch_loss = batch_loss + instance_loss
    batch_loss = batch_loss/len(query_features)

    return batch_loss

def cos_similarity(prototype,query_feature_contrastive, projection_head_model, hyperparameter_T):
    """Prototype of dim D, query_feature of Dimension D"""
    
    projected_query_feature = projection_head_model(query_feature_contrastive)
    cos = nn.CosineSimilarity()
    cosim = torch.exp(cos(prototype, projected_query_feature)/hyperparameter_T)
    
    return cosim


if __name__ =='__main__':
    prototypes = torch.rand(5,200)
    print(prototypes[1].size()) 

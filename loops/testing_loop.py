import torch
def prototypical_inference(sorted_query_features,prototypes):
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
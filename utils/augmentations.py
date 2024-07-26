import torch
from torchvision.transforms import v2

def image_augmentations(tensor_image):
    transform_horizontal_flip = v2.RandomHorizontalFlip(p=1.0)  # Always flip horizontally
    transform_vertical_flip = v2.RandomVerticalFlip(p=1.0)      # Always flip vertically
    transform_rotate_270 = v2.RandomRotation((270, 270))        # Always rotate by 270 degrees
    x_orig = tensor_image
    x_aug1 = transform_horizontal_flip(tensor_image)
    x_aug2 = transform_vertical_flip(tensor_image)
    x_aug3 = transform_rotate_270(tensor_image)
    x_list = [x_orig,x_aug1,x_aug2,x_aug3]
    return x_list

def audio_augmentations(audio_segment):
    '''Use a fixed number of audiomentations and return original audio and a fixed number of augmented audios'''
    pass
import PIL
import torch
import torch.nn as nn
import torchvision.transforms
import numpy as np


def crop_torch(img, target_aspect_ratio):
    tol = 1e-5
    img_height, img_width = img.shape[-2:]
    aspect_ratio = img_height / img_width
    if np.isclose(aspect_ratio, target_aspect_ratio, tol):
        return img
    if img_height > img_width:
        new_height = int(round(target_aspect_ratio*img_width))
        offset = (img_height - new_height)//2
        return img[..., offset:offset+new_height, :]
    new_width = int(round(img_height / target_aspect_ratio))
    offset = (img_width - new_width)//2
    return img[..., :, offset:offset+new_width]

def crop_pillow(img, target_aspect_ratio):
    # (left, upper, right, lower)
    tol = 1e-5
    img_width, img_height = img.size
    aspect_ratio = img_height / img_width
    if np.isclose(aspect_ratio, target_aspect_ratio, tol):
        return img
    if img_height > img_width:
        new_height = int(round(target_aspect_ratio*img_width))
        offset = (img_height - new_height)//2
        return img.crop((0, offset, img_width, offset+new_height))
    new_width = int(round(img_height / target_aspect_ratio))
    offset = (img_width - new_width)//2
    return img.crop((offset, 0, offset+new_width, img_height))

class CustomCropResize(nn.Module):
    
    def __init__(self, height, width):
        super().__init__()
        
        self.height = height
        self.width = width
        self.aspect_ratio = height / width
        
    def forward(self, img):
        if isinstance(img, torch.Tensor):
            img = crop_torch(img, self.aspect_ratio)
            img = torchvision.transforms.functional.resize(img, [self.height, self.width])
            # interpolation=bilinear, max_size=None, antialias=None
        elif isinstance(img, PIL.Image.Image):
            img = crop_pillow(img, self.aspect_ratio)
            img = img.resize((self.height, self.width), PIL.Image.ANTIALIAS)
        else:
            raise ValueError
        return img
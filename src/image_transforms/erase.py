# based on https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
import PIL.Image
import torchvision.transforms
import numpy as np

import torch
import torch.nn as nn

def erase(img, center_x, center_y, scale, log_ratio, value=0):
    #c, h, w = img.shape[-3:]
    if isinstance(img, PIL.Image.Image):
        w, h = img.size
    elif isinstance(img, (torch.Tensor, np.ndarray)):
        _, h, w = img.shape
    else:
        raise ValueError
    
    
    area = h*w
    #log_ratio = np.log(ratio)
    
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    
    erase_area = area*scale
    aspect_ratio = np.exp(log_ratio) # log uniform
    hh = int(round((erase_area*aspect_ratio)**0.5))
    ww = int(round((erase_area/aspect_ratio)**0.5))
    
    upper_left_x, upper_left_y = max(0, center_x - hh//2), max(0, center_y - ww//2)
    if not ((upper_left_x+hh < h) and (upper_left_y+ww < w)):
        return img
    return torchvision.transforms.functional.erase(
        img, upper_left_x, upper_left_y, hh, ww, value
    )

def random_erase(image, icoef_h_offset, icoef_w_offset, scale_bounds, log_ratio_bounds, rng):
    if isinstance(image, PIL.Image.Image):
        width, height = image.size
    elif isinstance(image, (torch.Tensor, np.ndarray)):
        channels, height, width = image.shape
    else:
        raise ValueError
    
    erase_h_offset = height // icoef_h_offset
    erase_w_offset = width // icoef_w_offset
    
    erase_center_y_bounds = (erase_h_offset, height-erase_h_offset)
    erase_center_x_bounds = (erase_w_offset, width-erase_w_offset)
    
    center_y = rng.integers(erase_center_y_bounds[0], erase_center_y_bounds[1], size=(1, ))[0]
    center_x = rng.integers(erase_center_x_bounds[0], erase_center_x_bounds[1], size=(1, ))[0]
    scale = rng.uniform(scale_bounds[0], scale_bounds[1], size=(1, ))[0]
    log_ratio = rng.uniform(log_ratio_bounds[0], log_ratio_bounds[1], size=(1, ))[0]
    return erase(image, center_x, center_y, scale, log_ratio, value=0)
        
class RandomErase(nn.Module):
    
    def __init__(
        self, icoef_h_offset, icoef_w_offset, scale_bounds, log_ratio_bounds, random_seed=0
    ):
        super(RandomErase, self).__init__()
        self.icoef_h_offset = icoef_h_offset
        self.icoef_w_offset = icoef_w_offset
        self.scale_bounds = scale_bounds
        self.log_ratio_bounds = log_ratio_bounds
        self.seed(random_seed)
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
        
    def forward(self, image):
        return random_erase(
            image, self.icoef_h_offset, self.icoef_w_offset,
            self.scale_bounds, self.log_ratio_bounds, self.rng
        )
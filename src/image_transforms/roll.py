import numpy as np
import torch.nn as nn
import torch
import torchvision
import PIL.Image

def roll_2d(image, hh, ww):
    if isinstance(hh, float):
        hh = int(round(hh))
    if isinstance(ww, float):
        ww = int(round(ww))
    if isinstance(image, PIL.Image.Image):
        image = torchvision.transforms.functional.to_tensor(image).roll((hh, ww), dims=(-2, -1))
        return torchvision.transforms.functional.to_pil_image(image)
    return image.roll((hh, ww), dims=(-2, -1))

def random_roll_2d(image, h, w, rng):
    #shape = image.shape
    if isinstance(h, int):
        assert h >= 0
        H = (-h, h)
    else:
        H = h
    if isinstance(w, int):
        assert w >= 0
        W = (-w, w)
    else:
        W = w
    hh = rng.integers(H[0], H[1], size=(1, ))
    ww = rng.integers(W[0], W[1], size=(1, ))
    return roll_2d(image, hh[0], ww[0])
        
class RandomRoll2d(nn.Module):
    
    def __init__(self, h, w, random_seed=0):
        super(RandomRoll2d, self).__init__()
        self.h = h
        self.w = w
        self.seed(random_seed)
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
        
    def forward(self, image):
        return random_roll_2d(image, self.h, self.w, self.rng)
    
class RandomRoll2d_icoef(nn.Module):
    
    def __init__(self, icoef_h_offset, icoef_w_offset, random_seed=0):
        super(RandomRoll2d_icoef, self).__init__()
        self.icoef_h_offset = icoef_h_offset
        self.icoef_w_offset = icoef_w_offset
        self.seed(random_seed)
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
        
    def forward(self, image):
        if isinstance(image, (torch.Tensor, np.ndarray)):
            _, height, width = image.shape
        else:
            width, height = image.size
        roll_h_offset = height // self.icoef_h_offset
        roll_w_offset = width // self.icoef_w_offset
        return random_roll_2d(image, roll_h_offset, roll_w_offset, self.rng)
    
    
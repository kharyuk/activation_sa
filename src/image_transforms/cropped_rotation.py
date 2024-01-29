import numbers
import numpy as np

import torch
import PIL
import torch.nn as nn
import torchvision.transforms.functional

# rotation without padding

# https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
# https://github.com/pytorch/vision/blob/fecd138504354010133f4503027d2506df8aa967/
# torchvision/transforms/transforms.py#L969-L977


##################

#F.pil_to_tensor(pic)

#F.to_pil_image(pic, mode=None)

###################

def crop_black_borders(image, angle):
    '''
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    '''
    height, width = image.size
    tan_a = np.abs(np.tan(angle*np.pi/180))
    coef = tan_a/(1-tan_a**2)
    width_margin = int(round(coef*(height-width*tan_a)))
    height_margin = int(round(coef*(width-height*tan_a)))
    #print(width_margin, height_margin)
    return image.crop((height_margin, width_margin, height-height_margin, width-width_margin))

def custom_rotation(img, angle, resample=False, expand=False, center=None):
    # img is a PIL object
    torch_tensor_flag = isinstance(img, torch.Tensor)
    if torch_tensor_flag:
        img = torchvision.transforms.functional.to_pil_image(img, mode=None)
    img_shape = img.size
    img = torchvision.transforms.functional.rotate(
        img, angle, resample, expand, center
    )
    img = crop_black_borders(img, angle)
    if torch_tensor_flag:
        img = torchvision.transforms.functional.pil_to_tensor(img)
    return torchvision.transforms.functional.resize(
        img, img_shape, torchvision.transforms.functional.InterpolationMode.BILINEAR#2 # bilinear
    )

class CustomRandomRotation(nn.Module):
    '''
    Rotate by a random angle and crop out black borders
    '''
    def __init__(self, degrees, resample=False, expand=False, center=None, random_seed=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        
        self.seed(random_seed)
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)

    #@staticmethod
    def get_params(self, degrees):
        angle = self.rng.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        return custom_rotation(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

import numbers

import torch
import torch.nn as nn
import torchvision.transforms.functional

import PIL
import numpy as np
import cv2
# https://stackoverflow.com/questions/46883245/blur-from-opencv-in-region-of-interest

def compute_elliptic_mask(width, height, a=1, b=1, mw=0, mh=0, phi=None):
    center = (int(round(height//2 + mw)), int(round(width//2 + mh)))
    axes = (int(round(a)), int(round(b)))
    #mask = np.zeros((width, height, 1), dtype=np.uint8)
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, -phi, 0, 360, (255, 255, 255), -1)
    return mask

def apply_masked_blur(image, mask, sigma=2.):
    blurred_image = np.array(
        image.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
    )
    image = np.array(image)
    return PIL.Image.fromarray(
        cv2.add(
            cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)),
            cv2.bitwise_and(blurred_image, blurred_image, mask=mask),
        )
    )

def proceed_nonnegative_or_sequenced_numeric_parameter(z):
    if isinstance(z, numbers.Number):
        if z < 0:
            raise ValueError('If it is a single number, the parameter must be positive.')
        return (0, z)
    if len(z) != 2:
        raise ValueError('If parameter is a sequence, it must be of length 2.')
    return z

def proceed_single_or_sequenced_numeric_parameter(z):
    if isinstance(z, numbers.Number):
        if z <= 0:
            raise ValueError('If it is a single number, the parameter must be positive.')
        return (-z, z)
    if len(z) != 2:
        raise ValueError('If parameter is a sequence, it must be of length 2.')
    return z

class RandomEllipticLocalBlur(nn.Module):
    def __init__(self, a, b, shift_x, shift_y, angle, sigma=2., random_seed=0):
        self.a = proceed_nonnegative_or_sequenced_numeric_parameter(a)
        self.b = proceed_nonnegative_or_sequenced_numeric_parameter(b)
        self.shift_x = proceed_single_or_sequenced_numeric_parameter(shift_x)
        self.shift_y = proceed_single_or_sequenced_numeric_parameter(shift_y)
        self.angle = proceed_single_or_sequenced_numeric_parameter(angle)
        
        self.sigma = sigma
        
        self.seed(random_seed)
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
    
    #@staticmethod
    def get_params(self, a_s, b_s, shift_x_s, shift_y_s, angle_s):
        a = self.rng.integers(*a_s)
        b = self.rng.integers(*b_s)
        shift_x = self.rng.integers(*shift_x_s)
        shift_y = self.rng.integers(*shift_y_s)
        angle = self.rng.uniform(*angle_s)
        return a, b, shift_x, shift_y, angle
    
    def __call__(self, image):
        torch_tensor_flag = isinstance(image, torch.Tensor)
        if torch_tensor_flag:
            image = torchvision.transforms.functional.to_pil_image(image, mode=None)
        width, height = image.size
        a, b, shift_x, shift_y, angle = self.get_params(
            self.a, self.b, self.shift_x, self.shift_y, self.angle
        )
        image = elliptic_local_blur(image, a, b, shift_x, shift_y, angle, self.sigma)
        if torch_tensor_flag:
            image = torchvision.transforms.functional.pil_to_tensor(image)
        return image
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '(angle={0})'.format(self.angle)
        format_string += ', a={0})'.format(self.a)
        format_string += ', b={0})'.format(self.b)
        format_string += ', shift_x={0})'.format(self.shift_x)
        format_string += ', shift_y={0})'.format(self.shift_y)
        format_string += ')'
        return format_string
    
#center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]

def elliptic_local_blur(image, a, b, shift_x, shift_y, angle, sigma=2.):
    # image = PIL object
    torch_tensor_flag = isinstance(image, torch.Tensor)
    if torch_tensor_flag:
        image = torchvision.transforms.functional.to_pil_image(image, mode=None)
    width, height = image.size
    mask = compute_elliptic_mask(width, height, a, b, shift_x, shift_y, angle)
    image = apply_masked_blur(image, mask, sigma)
    if torch_tensor_flag:
        image = torchvision.transforms.functional.pil_to_tensor(image)
    return image
    
def test_elliptic_local_blur(width, height):
    import matplotlib.pyplot as plt
    a = np.random.uniform(0, 1, size=(height, width))
    ima = PIL.Image.fromarray(a, 'L')

    plt.imshow(ima)
    plt.show()
    mask = compute_elliptic_mask(width, height, a=100, b=50, mw=10, mh=20, phi=20)
    plt.imshow(mask)
    plt.show()
    pima = apply_masked_blur(ima, mask, sigma=1.5)
    plt.imshow(pima)
    plt.show()
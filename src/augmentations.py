import sys
sys.path.append('../src/')

import torchvision.transforms

import torch_utils
import custom_compose
import image_transforms.roll
import image_transforms.cropped_rotation
import image_transforms.local_blur
import image_transforms.erase


##### Set 1
# 1) erase rectangular part of image ## scale=(0.02, 0.33), ratio=(0.3, 3.3)
erasing_transform = lambda img, center_x, center_y, scale, log_ratio: image_transforms.erase.erase(
    img, center_x, center_y, scale, log_ratio
)
# 2) adjust sharpness # 1.5, p=0.5
sharpness_transform = lambda img, sharpness_factor=1.5: torchvision.transforms.functional.adjust_sharpness(
    img, sharpness_factor
)
# 3) roll image
rolling_transform = lambda img, hh, ww: image_transforms.roll.roll_2d(img, hh, ww)
# 4) grayscale
grayscale_transform = lambda img, num_output_channels=3: torchvision.transforms.functional.rgb_to_grayscale(
    img, num_output_channels
)
# 5) gaussian blur
gaussian_blur_transform = lambda img, sigma, kernel_size=(7, 7): torchvision.transforms.functional.gaussian_blur(
    img, kernel_size, sigma
)

##### Set 2

# 1) brightness
brightness_transform = lambda img, brightness_factor: (
    torchvision.transforms.functional.adjust_brightness(
        img, brightness_factor
    )
)
# 2) contrast
contrast_transform = lambda img, contrast_factor: (
    torchvision.transforms.functional.adjust_contrast(
        img, contrast_factor
    )
)
# 3) saturation
saturation_transform = lambda img, saturation_factor: (
    torchvision.transforms.functional.adjust_saturation(
        img, saturation_factor
    )
)
# 4) hue
hue_transform = lambda img, hue_factor: (
    torchvision.transforms.functional.adjust_hue(
        img, hue_factor
    )
)                
# 5) horizontal flip
horizontal_flip_transform = lambda img: torchvision.transforms.functional.hflip(img)
# 6) custom rotation
custom_rotation_transform = lambda img, angle: (
    image_transforms.cropped_rotation.custom_rotation(img, angle)
)
# 7) elliptic local blur
elliptic_local_blur_transform = lambda img, a, b, shift_x, shift_y, angle: (
    image_transforms.local_blur.elliptic_local_blur(
        img, a, b, shift_x, shift_y, angle, sigma=3.
    )
)




import torchvision.transforms
from . import crop_resize
import torch_utils

_alexnet_pre_processing_functions = torchvision.transforms.Compose(
    [
        #image_transforms.crop_resize.CustomCropResize(desired_image_height, desired_image_width)
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)

_alexnet_post_processing_functions = torchvision.transforms.Compose(
    [
        #torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch_utils.img_normalize(x, unit=True)), # map to [0, 1]
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

_custom_cnn_pre_processing_functions = torchvision.transforms.Compose(
    [
        #image_transforms.crop_resize.CustomCropResize(desired_image_height, desired_image_width)
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)

_custom_cnn_post_processing_functions = torchvision.transforms.Compose(
    [
        #torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: torch_utils.img_normalize(x, unit=False)), # map to [0, 1]
        #torchvision.transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225]
        #)
    ]
)



_pre_processing_functions_dict = {
    'alexnet': _alexnet_pre_processing_functions,
    'vgg11': _alexnet_pre_processing_functions,
    'resnet18': _alexnet_pre_processing_functions,
    'custom': _alexnet_pre_processing_functions,
}
    
_post_processing_functions_dict = {
    'alexnet': _alexnet_post_processing_functions,
    'vgg11': _alexnet_post_processing_functions,
    'resnet18': _alexnet_post_processing_functions,
    'custom': _alexnet_post_processing_functions,
}
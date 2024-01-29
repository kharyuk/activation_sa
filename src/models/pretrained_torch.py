import sys
import os

import torch
import torchvision.models as models
from torch.utils import model_zoo

_family_names = [
    'ResNet', 'DenseNet', 'VGG', 'inception_v3', 'alexnet'
]
_singles = ['inception_v3', 'alexnet']

def findModelFamily(model_name):
    for family_name in _family_names:
        if family_name in _singles:
            if model_name == family_name:
                return family_name.lower()
        else:
            names = list(
                filter(
                    lambda x: x!= family_name,
                    getattr(models, family_name.lower()).__all__
                )
            )
            if model_name in names:
                return family_name.lower()
    return None
    
def getAvailableModels(family_name, verbose=True):
    if family_name.lower() == 'resnet':
        names = list(filter(lambda x: x != 'ResNet', models.resnet.__all__))
    elif family_name.lower() == 'densenet':
        names = list(filter(lambda x: x != 'DenseNet', models.densenet.__all__))
    elif family_name.lower() == 'vgg':
        names = list(filter(lambda x: x != 'VGG', models.vgg.__all__))
    elif (family_name.lower() == 'inception_v3') or (family_name.lower() == 'alexnet'):
        names = [family_name.lower()]
    else:
        raise ValueError
    if verbose:
        print(f'Available {family_name} (from torchvision): ' + ', '.join(names))
    return names

def model_loader(model_name, family_name, model_dirname, pretrained=True):
    if family_name is None:
        family_name = findModelFamily(model_name)
    else:
        family_name = family_name.lower()
        names = getAvailableModels(family_name, verbose=False)
        assert model_name in names
        
    model = getattr(models, model_name)()
    
    if pretrained:
        try:
            url = getattr(models, family_name).model_urls[model_name]
        except:
            url = sys.modules[
                f'torchvision.models.{family_name}'
            ].model_urls[model_name]
        model_filename = url.split('/')[-1]
        allFiles = os.listdir(model_dirname)
        if not (model_filename in allFiles):
            state_dict = model_zoo.load_url(url, model_dirname, progress=True)
        else:
            state_dict = torch.load(model_dirname+model_filename)
        model.load_state_dict(state_dict)
        del state_dict
    return model


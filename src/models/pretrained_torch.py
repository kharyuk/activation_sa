import sys
import os

import torch
import torchvision.models as models
from torch.utils import model_zoo

_family_names = [
    "ResNet", "DenseNet", "VGG", "inception_v3", "alexnet",
    "vision_transformer", "swin_transformer", "maxvit"
]
_singles = ["inception_v3", "alexnet"]

def findModelFamily(model_name):
    for family_name in _family_names:
        if family_name in _singles:
            if model_name == family_name:
                return family_name.lower()
            continue
        fname_lw = family_name.lower()
        if not hasattr(models, fname_lw):
            continue
        names = list(
            filter(
                lambda x: x != family_name,
                getattr(models, fname_lw).__all__
            )
        )
        if model_name in names:
            return fname_lw
    return None

def filter_results(models_list, base_class_name, filter_postfix):
    return list(
        filter(
            lambda x: x != base_class_name and not x.endswith(filter_postfix),
            models_list
        )
    )
    
def getAvailableModels(family_name, verbose=True):
    fname_lw = family_name.lower()
    if fname_lw == "resnet":
        names = filter_results(models.resnet.__all__, "ResNet", "Weights")
    elif fname_lw == "densenet":
        names = filter_results(models.densenet.__all__, "DenseNet", "Weights")
    elif fname_lw == "vgg":
        names = filter_results(models.vgg.__all__, "VGG", "Weights")
    elif fname_lw == "vision_transformer":
        names = filter_results(
            models.vision_transformer.__all__, "VisionTransformer", "Weights"
        )
    elif fname_lw == "swin_transformer":
        names = filter_results(
            models.swin_transformer.__all__, "SwinTransformer", "Weights"
        )
    elif fname_lw == "maxvit":
        names = filter_results(
            models.maxvit.__all__, "MaxVit", "Weights"
        )
    elif fname_lw in _singles:
        names = [fname_lw]
    else:
        raise ValueError
    if verbose:
        print(f"Available {family_name} (from torchvision): " + ", ".join(names))
    return names

def model_loader(
    model_name,
    family_name,
    model_dirname,
    model_filename=None,
    num_classes=1000, # imagenet default
    pretrained=True,
    device="cpu"
):
    try:
        if family_name is None:
            family_name = findModelFamily(model_name)
        else:
            family_name = family_name.lower()
            names = getAvailableModels(family_name, verbose=False)
            assert model_name in names
        model = getattr(models, model_name)(num_classes=num_classes)
    except:
        assert model_name == "vit_b_16"
        import vit16b
        model = vit16b.vit_b_16(num_classes=num_classes)
        
    
    
    if pretrained:
        if model_filename is None:
            try:
                url = getattr(models, family_name).model_urls[model_name]
            except:
                url = sys.modules[
                    f"torchvision.models.{family_name}"
                ].model_urls[model_name]
            model_filename = url.split("/")[-1]
            allFiles = os.listdir(model_dirname)
            if not (model_filename in allFiles):
                state_dict = model_zoo.load_url(url, model_dirname, progress=True)
            else:
                state_dict = torch.load(model_dirname+model_filename, map_location=device)
        else: # places365 fix
            #print("places365")
            model_path = os.path.join(model_dirname, model_filename)
            model_weights = torch.load(model_path, map_location=device)
            state_key = "state_dict"
            state_dict = {}
            if state_key in model_weights:
                for key, val in model_weights[state_key].items():
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = val
                del model_weights
            elif isinstance(model_weights, dict) and len(model_weights) > 0:
                state_dict = model_weights
            else:
                raise ValueError
        model.load_state_dict(state_dict)
        del state_dict
    return model


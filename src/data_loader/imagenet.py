######################################################################################
# https://pytorch.org/vision/stable/_modules/torchvision/datasets/imagenet.html
# # https://www.kaggle.com/c/imagenet-object-localization-challenge
# https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json

import os
import tarfile

from typing import Any, Dict, List, Iterator, Optional, Tuple
import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg

import pkgutil

_default_json_ind2class_file_path = './imagenet_classes_dict.json'
_default_dirname2class_mapping_filename = 'LOC_synset_mapping.txt'
_data_subpath = 'ILSVRC/Data/CLS-LOC'

def download_partial_imagenet(data_dirname, kaggle_username, kaggle_key, remove_targz=True):
    
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    # kaggle wants the (username, key) pair be specified before import
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    imagenet_dirname = 'imagenet'
    imagenet_dirname_path = os.path.join(data_dirname, imagenet_dirname)
    imagenet_filename = 'imagenet_object_localization_patched2019.tar.gz'

    api.competition_download_file(
        'imagenet-object-localization-challenge',
        file_name=imagenet_filename,
        path=imagenet_dirname_path
    )
    api.competition_download_file(
        'imagenet-object-localization-challenge',
        file_name='LOC_synset_mapping.txt',
        path=imagenet_dirname_path
    )

    imagenet_path = os.path.join(imagenet_dirname_path, imagenet_filename)
    
    # !tar -zxvf {imagenet_path} -C {imagenet_dirname_path} > imagenet_untar.log
    with tarfile.open(imagenet_path, 'r:gz') as targz_file:
        targz_file.extractall(path=imagenet_dirname_path)
    
    if remove_targz:
        os.remove(imagenet_path)
    return imagenet_dirname_path

def get_ind2class_dict(data_dirname, json_file_path=None):
    if json_file_path is None:
        json_file_path = _default_json_ind2class_file_path
    buf = pkgutil.get_data(__name__, json_file_path).decode()
    #with open(os.path.join(data_dirname, json_file_path), 'r') as json_file:
    #    buf = json_file.readlines()
    return eval(buf)

def get_dirname2class_dict(data_dirname, mapping_filename=None):
    if mapping_filename is None:
        mapping_filename = _default_dirname2class_mapping_filename
    dirname2class = {}
    with open(os.path.join(data_dirname, mapping_filename), 'r') as fd:
        buf = fd.readline().replace('\n', '')
        while len(buf) > 0:
            key, value = buf.split(' ', maxsplit=1)
            dirname2class[key] = value
            buf = fd.readline().replace('\n', '')
    return dirname2class
            
def revert_dict(dict_instance):
    return dict((value, key) for key, value in dict_instance.items())

class CustomImageNet(ImageFolder):
    '''
        Customized ImageNet dataloader. Works with Kaggle sub-dataset.
        imagenet-object-localization-challenge
    '''

    def __init__(
        self, root: str, split: str='train', download: Optional[str]=None,
        **kwargs: Any
    ) -> None:
        self.root = os.path.join(root, _data_subpath)
        
        self.ind2class = get_ind2class_dict('')#data_dirname)
        self.dirname2class = get_dirname2class_dict(root) #wnid_to_classes
        
        self.class2ind = revert_dict(self.ind2class)
        self.class2dirname = revert_dict(self.dirname2class)
        
        self.classes = [self.ind2class[i] for i in range(len(self.ind2class))] # wnids
        
        self.split = verify_str_arg(split, 'split', ('train', 'val'))
        
        super(CustomImageNet, self).__init__(self.split_folder, **kwargs)
        
        self.root = os.path.join(root, _data_subpath)
        
        #self.wnids = self.classes
        self.wnids = [self.class2dirname[self.ind2class[i]] for i in range(len(self.ind2class))] # wnids
        #self.wnid_to_idx = self.class_to_idx
        self.wnid_to_idx = self.class2ind
        #self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.classes = [self.ind2class[i] for i in range(len(self.ind2class))]
        self.class_to_idx = self.class2ind #{cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)




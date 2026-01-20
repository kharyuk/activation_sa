################################################################################
# https://pytorch.org/vision/stable/_modules/torchvision/datasets/imagenet.html


import os
import zipfile

from typing import Any, Dict, List, Iterator, Optional, Tuple
import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg

import pkgutil

_default_json_ind2class_file_path: str = "./places365_classes_dict.json"
_places365_dataset_name: str = "benjaminkz/places365"

def download_partial_places365(
    data_dirname: str,
    kaggle_username: str,
    kaggle_key: str,
    remove_targz: bool = True
) -> str:
    
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    # kaggle wants the (username, key) pair be specified before import
    from kaggle.api.kaggle_api_extended import KaggleApi

    api: KaggleApi = KaggleApi()
    api.authenticate()

    places365_dirname: str = "places365"
    places365_dirname_path: str = os.path.join(
        data_dirname, places365_dirname
    )
    places365_filename: str = "places365.zip"

    api.dataset_download_files(
        _places365_dataset_name,
        path=places365_dirname_path
    )
    
    places365_path: str = os.path.join(
        places365_dirname_path, places365_filename
    )
    with zipfile.ZipFile(places365_path, 'r') as zip_file:
        zip_file.extractall(path=places365_dirname_path)
    
    if remove_targz:
        os.remove(places365_path)
    
    return places365_dirname_path

def get_ind2class_dict(
    data_dirname: str,
    json_file_path: Optional[str] = None
) -> Dict[int, str]:
    if json_file_path is None:
        json_file_path = _default_json_ind2class_file_path
    buf: str = pkgutil.get_data(__name__, json_file_path).decode()
    #with open(os.path.join(data_dirname, json_file_path), 'r') as json_file:
    #    buf = json_file.readlines()
    return eval(buf)
            
def revert_dict(dict_instance: Dict[Any, Any]) -> Dict[Any, Any]:
    return dict((value, key) for key, value in dict_instance.items())

def map_class_to_dirname(class_str: str) -> str:
    return class_str.replace(' ', '-')

class CustomPlaces365(ImageFolder):
    '''
        Customized Places365 dataloader. Works with Kaggle upload.
    '''

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.root: str = root
        
        self.ind2class: Dict[int, str] = get_ind2class_dict("./")
        self.class2ind: Dict[str, int] = revert_dict(self.ind2class)
        # wnids
        self.classes: list[str] = [
            self.ind2class[i] for i in range(len(self.ind2class))
        ]
        self.split: str = verify_str_arg(split, "split", ("train", "val"))
        
        super(CustomPlaces365, self).__init__(self.split_folder, **kwargs)
        
        #self.wnids = self.classes
        self.wnids: list[str] = [
            map_class_to_dirname(class_str) for class_str in self.classes
        ] # wnids
        #self.wnid_to_idx = self.class_to_idx
        self.wnid_to_idx: Dict[str, int] = self.class2ind
        #self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx: Dict[str, int] = self.class2ind
        #{cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)




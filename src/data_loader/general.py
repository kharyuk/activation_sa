import os

import numpy as np
import torch
import torchvision.transforms

import torch_learning
import torch_utils

#def normalize(image, eps=1e-10, unit=False):
#    image = image - image.min(dim=1, keepdim=True).values
#    image = image / (image.max(dim=1, keepdim=True).values + eps)
#    if unit:
#        return image
#    return (image - 0.5) / 0.5
#
#_basic_transforms = torchvision.transforms.Compose(
#    [
#        torchvision.transforms.ToTensor(),
#        torchvision.transforms.Lambda(normalize)
#    ]
#)

def get_dataset_attr(dataset, attr_name):
    if hasattr(dataset, attr_name):
        return getattr(dataset, attr_name)
    return getattr(dataset.dataset, attr_name) # Subset of dataset



def load_single_dataset_part(
    dataset_loader,
    dataset_loader_kwargs_dict,
    data_path,
    batch_size=None,
    num_workers=0,
    augmentation_set=None,
    subset_size=None,
    samples_per_class=None,
    shuffle=False,
    splitting_random_state=0,
    return_torch_dataset=True,
    return_torch_loader=False,
    return_plain_data=False,
    labels_attr_name='labels',
    dataset_init_split='train',
    do_pretensorize=False,
    do_posttensorize=False,
    do_normalize=False,
    unit_normalize=True
):
    #data_dirname = '../data/'
    #stl10_subdirname = 'stl10'
    #data_path = os.path.join(data_dirname, stl10_subdirname)
    assert return_torch_dataset or return_torch_loader or return_plain_data
    assert not (do_pretensorize and do_posttensorize)
    
    os.makedirs(data_path, exist_ok=True)
    
    dataloader, Xy_data = None, None

    c_transforms = []
    if do_pretensorize:
        c_transforms.append(torchvision.transforms.ToTensor())
    if augmentation_set is not None:
        c_transforms.append(augmentation_set)
    if do_posttensorize:
        c_transforms.append(torchvision.transforms.ToTensor())
    if do_normalize:
        c_transforms.append(
            torchvision.transforms.Lambda(
                lambda image: torch_utils.img_normalize(image, eps=1e-20, unit=unit_normalize)
            )
        )
    c_transforms = torchvision.transforms.Compose(c_transforms)
        
    dataset_loader_kwargs_dict['split'] = dataset_loader_kwargs_dict.get(
        'split', dataset_init_split
    )

    dataset = dataset_loader(
        root=data_path,
        transform=c_transforms,
        **dataset_loader_kwargs_dict
    )
    num_classes = len(dataset.classes)

    if (subset_size is None) and (samples_per_class is None):
        if return_plain_data:
            Xy_data = dataset.data, getattr(dataset, labels_attr_name)
    else:
        # splitting it into train and valid parts
        p1_dataset, subset_indices = torch_learning.extract_class_equal_sampled_dataset(
            dataset,
            samples_per_class,
            n_samples_all=subset_size,
            labels_attr_name=labels_attr_name,
            random_seed=splitting_random_state
        )
        if return_plain_data:
            Xy_data = (
                p1_dataset.data[subset_indices], getattr(dataset, labels_attr_name)[subset_indices]
            )
        if return_torch_dataset:
            dataset = p1_dataset
    
    if return_torch_loader:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers
        )
    rv = []
    if return_torch_dataset:
        rv.append(dataset)
    if return_torch_loader:
        rv.append(dataloader)
    if return_plain_data:
        rv.append(Xy_data)
    return tuple(rv), num_classes

def load_single_part_bipartitioned_dataset(
    dataset_loader,
    dataset_loader_kwargs_dict,
    data_path,
    batch_size=None,
    num_workers=0,
    augmentation_set_partition1=None,
    augmentation_set_partition2=None,
    n_samples_per_class=None,
    n_samples_all=None,
    train_size=0.95,
    valid_size=None,
    split_classwise=True,
    equating_random_state=0,
    splitting_random_state=0,
    return_torch_dataset=True,
    return_torch_loader=False,
    return_plain_data=False,
    labels_attr_name='labels',
    dataset_init_split='train',
    do_pretensorize=False,
    do_posttensorize=False,
    do_normalize=False,
    unit_normalize=True,
    shuffle=False
):
    
    assert (train_size is not None) or (valid_size is not None)
    assert not (do_pretensorize and do_posttensorize)
    
    partition1_c_transforms = []
    if do_pretensorize:
        partition1_c_transforms.append(torchvision.transforms.ToTensor())
    if augmentation_set_partition1 is not None:
        partition1_c_transforms.append(augmentation_set_partition1)
    if do_posttensorize:
        partition1_c_transforms.append(torchvision.transforms.ToTensor())
    if do_normalize:
        partition1_c_transforms.append(
            torchvision.transforms.Lambda(
                lambda image: torch_utils.img_normalize(image, eps=1e-20, unit=unit_normalize)
            )
        )    
    partition1_c_transforms = torchvision.transforms.Compose(partition1_c_transforms)
    
    partition2_c_transforms = []
    if do_pretensorize:
        partition2_c_transforms.append(torchvision.transforms.ToTensor())
    if augmentation_set_partition2 is not None:
        partition2_c_transforms.append(augmentation_set_partition2)
    if do_posttensorize:
        partition2_c_transforms.append(torchvision.transforms.ToTensor())
    if do_normalize:
        partition2_c_transforms.append(
            torchvision.transforms.Lambda(
                lambda image: torch_utils.img_normalize(image, eps=1e-20, unit=unit_normalize)
            )
        )    
    partition2_c_transforms = torchvision.transforms.Compose(partition2_c_transforms)
    
    dataset_loader_kwargs_dict['split'] = dataset_loader_kwargs_dict.get(
        'split', dataset_init_split
    )
        
    partition1_dataset = dataset_loader(
        root=data_path,
        transform=partition1_c_transforms,
        **dataset_loader_kwargs_dict
    )
    partition2_dataset = dataset_loader(
        root=data_path,
        transform=partition2_c_transforms,
        **dataset_loader_kwargs_dict
    )
    num_classes = len(partition1_dataset.classes)
    if (n_samples_per_class is not None) or (n_samples_all is not None):
        partition1_dataset, subset_indices = torch_learning.extract_class_equal_sampled_dataset(
            partition1_dataset,
            n_samples_per_class,
            n_samples_all,
            labels_attr_name=labels_attr_name,
            random_seed=equating_random_state
        )
        partition2_dataset = torch.utils.data.Subset(partition2_dataset, subset_indices)

    # splitting it into train and valid parts
    (partition1_indices, partition2_indices),  = torch_learning.split_dataset_train_valid(
        partition1_dataset, train_size=train_size, valid_size=valid_size,
        random_state=splitting_random_state, labels_attr_name=labels_attr_name,
        return_datasets=False, return_indices=True, split_classwise=split_classwise
    )
    if (n_samples_per_class is not None) or (n_samples_all is not None):
        partition1_indices = subset_indices[partition1_indices]
        partition2_indices = subset_indices[partition2_indices]
    targets = np.array(get_dataset_attr(partition1_dataset, labels_attr_name))
    #train_labels = np.array(getattr(partition1_dataset, labels_attr_name))[partition1_indices]
    #valid_labels = np.array(getattr(partition2_dataset, labels_attr_name))[partition2_indices]

    if return_plain_data:
        Xy_data_partition1 = (
            get_dataset_attr(partition1_dataset, 'data')[partition1_indices],
            targets[partition1_indices].tolist()
        )
        Xy_data_partition2 = (
            get_dataset_attr(partition2_dataset, 'data')[partition2_indices],
            targets[partition2_indices].tolist()
        )
        
    if return_torch_dataset:
        if isinstance(partition1_dataset, torch.utils.data.Subset):
            partition1_dataset.indices = partition1_indices
            partition2_dataset.indices = partition2_indices
        else:
            partition1_dataset = torch.utils.data.Subset(partition1_dataset, partition1_indices)
            #setattr(train_dataset, labels_attr_name, train_labels)
            partition2_dataset = torch.utils.data.Subset(partition2_dataset, partition2_indices)
            #setattr(valid_dataset, labels_attr_name, valid_labels)
    
    #train_loader_random_generator = torch.Generator()
    if return_torch_loader:
        partition1_dataloader = torch.utils.data.DataLoader(
            dataset=partition1_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            #generator=train_loader_random_generator
        )
        partition2_dataloader = torch.utils.data.DataLoader(
            dataset=partition2_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers
        )
    rv_partition1, rv_partition2 = [], []
    if return_torch_dataset:
        rv_partition1.append(partition1_dataset)
        rv_partition2.append(partition2_dataset)
    if return_torch_loader:
        rv_partition1.append(partition1_dataloader)
        rv_partition2.append(partition2_dataloader)
    if return_plain_data:
        rv_partition1.append(Xy_data_partition1)
        rv_partition2.append(Xy_data_partition2)
    return tuple(rv_partition1), tuple(rv_partition2), num_classes

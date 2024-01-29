import os
import time
import logging 

import sklearn.model_selection
import torch
import torch.nn as nn
import torch.autograd
import numpy as np

import torch.utils.data

def extract_class_equal_sampled_dataset(
    dataset,
    n_samples_per_class=None,
    n_samples_all=None,
    labels_attr_name='labels',
    random_seed=0,
    sort_ind=True
):
    assert (n_samples_per_class is None) != (n_samples_all is None)
    n_classes = len(dataset.classes)
    n_samples_available = len(dataset)
    if isinstance(dataset, torch.utils.data.Subset):
        targets = np.array(getattr(dataset.dataset, labels_attr_name))
        indices = np.array(dataset.indices)
        targets = targets[indices]
    else:
        targets = np.array(getattr(dataset, labels_attr_name))
    
    classes_enc, n_samples_per_class_available = np.unique(targets, return_counts=True)
    min_within_class_samples = n_samples_per_class_available.min()
    if n_samples_per_class is None:
        n_samples_per_class = n_samples_all // n_classes
    else:
        n_samples_all = n_samples_per_class*n_classes
    assert n_classes <= n_samples_all <= n_samples_available
    assert n_samples_per_class <= min_within_class_samples
    
    rng = np.random.default_rng(random_seed)
    final_ind = np.empty(0, dtype='i')
    for cur_class in range(n_classes):
        ind = np.where(targets == cur_class)[0]
        ind = rng.permutation(ind)
        final_ind = np.append(final_ind, ind[:n_samples_per_class], axis=-1)
    if sort_ind:
        final_ind = np.sort(final_ind)
    return torch.utils.data.Subset(dataset, final_ind), final_ind

    
def split_dataset_train_valid(
    dataset,
    train_size=0.9,
    valid_size=None,
    random_state=None,
    labels_attr_name='labels',
    return_datasets=True,
    return_indices=False,
    split_classwise=True
):
    n_samples = len(dataset)
    if isinstance(dataset, torch.utils.data.Subset):
        targets = np.array(getattr(dataset.dataset, labels_attr_name))
        yind = np.array(dataset.indices)
        targets = targets[yind]
    else:
        targets = np.array(getattr(dataset, labels_attr_name))
    
    if split_classwise:
        def check_size(size1, max_size):
            assert size1 >= 0
            if isinstance(size1, float):
                assert size1 < 1
                size2 = 1-size1
            elif isinstance(size1, int):
                assert size1 <= max_size
            else:
                raise ValueError
            return size2
        assert (train_size is not None) or (valid_size is not None)
        classes, samples_per_classes = np.unique(targets, return_counts=True)
        min_samples_per_class = samples_per_classes.min()
        size_int_flag = False
        if (train_size is not None) and (valid_size is not None):
            assert type(train_size) == type(valid_size)
            assert train_size >= 0
            assert valid_size >= 0
            size_sum = train_size + valid_size
            if isinstance(train_size, int):
                assert 0 < size_sum <= min_samples_per_class
                size_int_flag = True
            elif isinstance(train_size, float):
                assert 0 < size_sum <= 1
            raise ValueError
        elif (train_size is None):
            train_size = check_size(valid_size, min_samples_per_class)
            size_int_flag = isinstance(valid_size, int)
        else:
            valid_size = check_size(train_size, min_samples_per_class)
            size_int_flag = isinstance(train_size, int)
        rng = np.random.default_rng(seed=random_state)
    
        train_ind, valid_ind = np.empty(0, dtype='i'), np.empty(0, dtype='i')
        for i, cl_i in enumerate(classes):
            ind = np.where(targets == cl_i)[0]
            ind = rng.permutation(ind)
            if size_int_flag:
                if train_ind is None:
                    train_ind = samples_per_classes[i]-valid_ind
                elif valid_ind is None:
                    valid_ind = samples_per_classes[i]-train_ind
                train_ind = np.append(train_ind, ind[:train_size], axis=-1)
                valid_ind = np.append(valid_ind, ind[train_size:train_size+valid_size], axis=-1)
            else:
                train_size_int = int(round(samples_per_classes[i]*train_size))
                valid_size_int = int(round(samples_per_classes[i]*valid_size))
                train_ind = np.append(train_ind, ind[:train_size_int], axis=-1)
                valid_ind = np.append(valid_ind, ind[train_size_int:train_size_int+valid_size_int], axis=-1)
        train_ind = np.sort(train_ind)
        valid_ind = np.sort(valid_ind)
    else:
        train_valid_splitter = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=valid_size, train_size=train_size,
            random_state=random_state
        )
        (train_ind, valid_ind), = train_valid_splitter.split(
            np.empty([n_samples, 0]), targets
        )
    rv = []
    if return_datasets:
        if isinstance(dataset, torch.utils.data.Subset):
            train_dataset = copy.copy(dataset)
            valid_dataset = copy.copy(dataset)
            try:
                setattr(train_dataset, labels_attr_name, targets[train_ind])
                setattr(valid_dataset, labels_attr_name, targets[valid_ind])
            except:
                pass
            train_dataset.indices = train_dataset.indices[train_ind]
            valid_dataset.indices = valid_dataset.indices[valid_ind]
        else:
            train_dataset = torch.utils.data.Subset(dataset, train_ind)
            valid_dataset = torch.utils.data.Subset(dataset, valid_ind)
        rv.append((train_dataset, valid_dataset))
    if return_indices:
        rv.append((train_ind, valid_ind))
    return tuple(rv)

def initialize_logging(log_path):
    for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path, filemode='w+',# encoding='utf-8',
        level=logging.INFO, format='%(asctime)s %(message)s',
        datefmt='%d/%m %H:%M:%S'
    )
    
def get_ytrue_ypred(network, dataloader):
    '''
    we need ytrue in case of shuffle=True during data loading
    '''
    y_true_list, y_pred_list = [], []
    for i, (partX, partY) in enumerate(dataloader):
        ypred = network(partX, logits=False)
        y_true_list.append(partY.detach().numpy())
        y_pred_list.append(ypred.detach().numpy())
    return np.concatenate(y_true_list, axis=0), np.concatenate(y_pred_list, axis=0)

def get_ytrue_ypred_augmented(
    network,
    dataloader,
    augmentation_transforms,
    preprocessing_transforms=None,
    postprocessing_transforms=None,
    Naug=1
):
    '''
    we need ytrue in case of shuffle=True during data loading
    '''
    y_true_list, y_pred_list = [], []
    for i, (partX, partY) in enumerate(dataloader):
        if preprocessing_transforms is not None:
            partX = preprocessing_transforms(partX)
        local_ypred_list = []
        for j in range(Naug):
            copy_partX = augmentation_transforms(partX.clone())
            if postprocessing_transforms is not None:
                copy_partX = postprocessing_transforms(copy_partX)
            ypred = network(copy_partX, logits=False)
            local_ypred_list.append(ypred.detach().numpy())
        local_ypred_list = np.array(local_ypred_list)
        
        y_true_list.append(partY.detach().numpy())
        y_pred_list.append(np.mean(local_ypred_list, axis=0))
    return np.concatenate(y_true_list, axis=0), np.concatenate(y_pred_list, axis=0)

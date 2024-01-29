# regular + augmented

#### 0. parsing the input arguments ##########################################################
import argparse

# (manual) how to pass dicts without str/eval
# https://gist.github.com/vadimkantorov/37518ff88808af840884355c845049ea#file-argparse_dict_argument-py
# but we will use str/eval workaround


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--do_compute_activations',
    type=int,
    default=1
)
arg_parser.add_argument(
    '--do_compute_values',
    type=int,
    default=1
)
arg_parser.add_argument(
    '--recompute_activations',
    type=int,
    default=0
)


arg_parser.add_argument(
    '--batch_size_activations_computing',
    type=int,
    default=500,
    help='specifies the size of mini-batch while compiting the activations'
)

arg_parser.add_argument(
    '--samples_per_class_train',
    type=int,
    default=732
)
arg_parser.add_argument(
    '--samples_per_class_valid',
    type=int,
    default=50
)
arg_parser.add_argument(
    '--subset_random_state_train',
    type=int,
    default=753
)
arg_parser.add_argument(
    '--subset_random_state_valid',
    type=int,
    default=753
)

arg_parser.add_argument(
    '--buffer_size_shpv_computing',
    type=int,
    default=1000,
    help='specifies the buffer size of activations per process while compuing the Shapley values'
)


arg_parser.add_argument(
    '--torch_seed',
    type=int,
    default=817,
    help='a random seed value for torch'
)
arg_parser.add_argument(
    '--numpy_seed',
    type=int,
    default=715,
    help='a random seed value for numpy'
)
arg_parser.add_argument(
    '--class_sampler_seed',    
    type=int,
    default=512,
    help='random state for class variable sampler'
)


arg_parser.add_argument(
    '--augmentation_set_number',
    type=int,
    default=1,
    help='1: set1; 2: set2; 3: set1+set2'
)

arg_parser.add_argument(
    '--dataset_part',
    type=str,
    default='train',
    help='train / valid / test: specifies the part of dataset to be used in experiment'
)
arg_parser.add_argument(
    '--activations_dirname',
    type=str,
    help='path to directory for storing the computed values'
)
arg_parser.add_argument(
    '--activations_fnm_prefix',
    type=str,
    default='shpv_alexnet_imagenet_ILSVRC_activations',
    help='name prefix for hdf5 file containing the activations'
)
arg_parser.add_argument(
    '--values_fnm_prefix',
    type=str,
    default='shpv_alexnet_imagenet_ILSVRC_values',
    help='name prefix for hdf5 file containing the computed values'
)


arg_parser.add_argument(
    '--Nouter_samples',
    type=int,
    default=2**6,
    help='Number of samples to estimate Shapley values [outer loop]'
)
arg_parser.add_argument(
    '--Ninner_samples',
    type=int,
    default=2**4,
    help='Number of samples to estimate Shapley values [inner loop]'
)
arg_parser.add_argument(
    '--Npermutations',
    type=int,
    default=2**8,
    help='Number of permutations to estimate Shapley values.'
)


arg_parser.add_argument(
    '--permutation_random_seed',
    type=int,
    default=256,
    help='Random seed for sampling permutatons'
)
arg_parser.add_argument(
    '--problem_sampler_seed',
    type=int,
    default=128,
    help='Random seed for sampling problem parameters'
)


arg_parser.add_argument(
    '--Njobs_shpv_computing',
    type=int,
    default=2,
    help='Number of parallel processes used to compute Shapley values'        
)



arg_parser.add_argument(
    '--os_environment_config',
    type=str,
    default='',
    help=(
        'additional setup for environment which will be processed as calling '
        'os.environ[key] = value (as python dictionary converted to string). '
        'E.g., (CUDA_DEVICE_ORDER, PCI_BUS_ID), (CUDA_VISIBLE_DEVICES, 2) '
    )
)
arg_parser.add_argument(
    '--mkl_num_threads',
    type=int,
    default=4,
    help='limits the number of MKL threads be occupied by the script'
)
arg_parser.add_argument(
    '--data_dirname',
    type=str,
    help='path to directory with dataset'
)
arg_parser.add_argument(
    '--device',
    type=str,
    default='cpu',
    help='Specify the device for torch (cpu/cuda:X)'
)
arg_parser.add_argument(
    '--model_dirname',
    type=str,
    help='path to directory containing the trained weights'
)



arg_parser.add_argument(
    '--augset1_p',
    type=float,
    default=0.5,
    help='probability of applying single transform from a set1'
)
arg_parser.add_argument(
    '--augset2_p',
    metavar='augset2_p',
    type=float,
    default=0.5,
    help='probability of applying single transform from a set2'
)
arg_parser.add_argument(
    '--libstdcpp_path',
    type=str,
    default=''
)
arg_parser.add_argument(
    '--remove_activations_hdf5',
    type=int,
    default=0
)
arg_parser.add_argument(
    '--desired_image_height',
    type=int,
    default=224
)
arg_parser.add_argument(
    '--desired_image_width',
    type=int,
    default=224
)

arg_parser.add_argument(
    '--network_name',
    type=str
)
arg_parser.add_argument(
    '--network_modules',
    type=str
)
arg_parser.add_argument(
    '--classification_layer_name',
    type=str
)

arg_parser.add_argument(
    '--class_selector_seed',
    type=int,
    default=101,
    help='random state for class variable selection'
)
arg_parser.add_argument(
    '--partition_sampler_seed',
    type=int,
    default=102,
    help='random state for partition variable sampling'
)

arg_parser.add_argument(
    '--use_permutation_variable',
    type=int,
    default=0
)
arg_parser.add_argument(
    '--use_partition_variable',
    type=int,
    default=0
)

arg_parser.add_argument(
    '--variables_permutation_random_seed',
    type=int,
    default=190
)


args = arg_parser.parse_args()

do_compute_activations = bool(args.do_compute_activations)
do_compute_values = bool(args.do_compute_values)
recompute_activations = bool(args.recompute_activations)

torch_seed = args.torch_seed
numpy_seed = args.numpy_seed
mkl_num_threads = args.mkl_num_threads
libstdcpp_path = args.libstdcpp_path
os_environment_config = None
if (args.os_environment_config is not None) and (len(args.os_environment_config) > 0):
    os_environment_config = eval(args.os_environment_config)
    
# at first, we parse the arguments that control the hardware/software configuration
import os
# additional config
if os_environment_config is not None:
    for key in os_environment_config:
        if key in os.environ:
            os.environ[key] += f':{os_environment_config[key]}'
        else:
            os.environ[key] = os_environment_config[key]

import ctypes
if len(libstdcpp_path) > 0:
    _stdcxx_lib = ctypes.cdll.LoadLibrary(libstdcpp_path)

# make the src directory visible
import sys
sys.path.append('../src/')

# set limitations on hardware
import hardware_setup
hardware_setup.mkl_set_num_threads(num_threads=mkl_num_threads)

import software_setup
import numpy as np
import torch

np.random.seed(numpy_seed)
software_setup.torch_init()
software_setup.torch_seeding(torch_seed, random_seed=None)
#software_setup.setup_multiprocessing()

# now, continue with the others
import exp_assistance

samples_per_class_train = args.samples_per_class_train
samples_per_class_valid = args.samples_per_class_valid
subset_random_state_train = args.subset_random_state_train
subset_random_state_valid = args.subset_random_state_valid

augmentation_set_number = args.augmentation_set_number
dataset_part = args.dataset_part #train_val_test
#assert dataset_part in ['train', 'valid', 'test']

activations_dirname = args.activations_dirname
activations_fnm_prefix = args.activations_fnm_prefix
values_fnm_prefix = args.values_fnm_prefix

class_sampler_seed = args.class_sampler_seed


batch_size_activations_computing = args.batch_size_activations_computing

buffer_size_shpv_computing = args.buffer_size_shpv_computing
Njobs_shpv_computing = args.Njobs_shpv_computing

#os_environment_config = None
#if (args.os_environment_config is not None) and (len(args.os_environment_config) > 0):
#    os_environment_config = eval(args.os_environment_config)
data_dirname = args.data_dirname
device = args.device
model_dirname = args.model_dirname

augset1_p = args.augset1_p
augset2_p = args.augset2_p

remove_activations_hdf5 = bool(args.remove_activations_hdf5)

Nouter_samples = args.Nouter_samples
Ninner_samples = args.Ninner_samples
Npermutations = args.Npermutations
permutation_random_seed = args.permutation_random_seed
problem_sampler_seed = args.problem_sampler_seed

desired_image_height = args.desired_image_height
desired_image_width = args.desired_image_width

use_permutation_variable = bool(args.use_permutation_variable)
use_partition_variable = bool(args.use_partition_variable)

network_name = args.network_name
network_modules = eval(args.network_modules)
classification_layer_name = args.classification_layer_name

class_selector_seed = args.class_selector_seed
partition_sampler_seed = args.partition_sampler_seed

variables_permutation_random_seed = args.variables_permutation_random_seed


#######################################################################################################
### 1. loading all necessary packages

import time

import torch.nn as nn
import torchvision

import sklearn.metrics

import h5py

import torch_utils
import model_dissection

import models.pretrained_torch
import data_loader.general
import data_loader.imagenet


import sensitivity_analysis.shapley
import sensitivity_analysis.build_problem_shpv

import image_transforms.pre_post_processing

######################################################
#### 2. loading ILSVRC dataset

#dataset_loader_kwargs_dict = {} # no options
#dataset = data_loader.imagenet.CustomImageNet('../data/imagenet/', 'train')

batch_size = None
num_workers = 0
labels_attr_name = 'targets'
image_shape = (3, desired_image_height, desired_image_width)

dataset_train = None
dataset_valid = None
dataset_test = None
len_dataset = 0
if 'train' in dataset_part:
    (dataset_train, ), num_classes = data_loader.general.load_single_dataset_part(
        data_loader.imagenet.CustomImageNet,
        {},
        data_dirname,
        batch_size,
        num_workers=num_workers,
        augmentation_set=None,
        subset_size=None,
        samples_per_class=samples_per_class_train,
        shuffle=False,
        splitting_random_state=subset_random_state_train,
        return_torch_dataset=True,
        return_torch_loader=False,
        return_plain_data=False,
        labels_attr_name=labels_attr_name,
        dataset_init_split='train',
        do_pretensorize=False,
        do_normalize=False
    )
    len_dataset += 1
if 'valid' in dataset_part:
    (dataset_valid, ), num_classes = data_loader.general.load_single_dataset_part(
        data_loader.imagenet.CustomImageNet,
        {},
        data_dirname,
        batch_size,
        num_workers=num_workers,
        augmentation_set=None,
        subset_size=None,
        samples_per_class=samples_per_class_valid,
        shuffle=False,
        splitting_random_state=subset_random_state_valid,
        return_torch_dataset=True,
        return_torch_loader=False,
        return_plain_data=False,
        labels_attr_name=labels_attr_name,
        dataset_init_split='val',
        do_pretensorize=False,
        do_normalize=False
    )
    len_dataset += 1
if 'test' in dataset_part:
    raise NotImplementedError
    #(dataset, ), image_shape, num_classes = data_loader.general.load_test(
    #    data_loader.imagenet.CustomImageNet, dataset_loader_kwargs_dict, data_dirname, batch_size,
    #    num_workers, augmentation_set_test=None, subset_size=subset_size, shuffle=False, 
    #    splitting_random_state=subset_random_state, return_torch_dataset=True,
    #    return_torch_loader=False, return_plain_data=False, labels_attr_name=labels_attr_name
    #)
#else:
#    raise ValueError

dataset = []
if dataset_train is not None:
    dataset.append(dataset_train)
if dataset_valid is not None:
    dataset.append(dataset_valid)
if dataset_test is not None:
    dataset.append(dataset_test)

if len_dataset == 1:
    dataset = dataset[0]
    
######################################################
#### 3. compose model and load the trained weights

network = models.pretrained_torch.model_loader(
    model_name=network_name, family_name=None, model_dirname=model_dirname, pretrained=True
)

parameters_number = sum(p.numel() for p in  network.parameters()) # mem_all[-1]
print(f'Number of parameters: {parameters_number}')
network.eval();

def softmax_hook(name, module, inputs, output):
    return output.softmax(dim=-1)

softmax_hook_handlers = torch_utils.register_forward_hook_by_name(
    network, softmax_hook, classification_layer_name # classifier.6 is the last Linear layer without non-linearity
)
inference_params = {}

pre_processing_functions = image_transforms.pre_post_processing._pre_processing_functions_dict[
    network_name
]
post_processing_functions = image_transforms.pre_post_processing._post_processing_functions_dict[
    network_name
]




######################################################
#### 4. select layers

# extract module names which activations are to be analysed
#network_leaves = model_dissection.get_flatten_leaves(model)
filtered_network_leaves = list(network_modules)
print('Layers to be analyzed:')
print(filtered_network_leaves)

######################################################
#### 5. build a problem configuration

(
    problem, transform_functions_list, Vi_groups, Vtheta0_dict
) = sensitivity_analysis.build_problem_shpv.build_augmented_classification_problem(
    augmentation_set_number,
    image_shape,
    num_classes,
    p_aug_set1=augset1_p,
    p_aug_set2=augset2_p,
    use_permutation_variable=use_permutation_variable,
    use_partition_variable=use_partition_variable
)
sampler = sensitivity_analysis.shapley.CustomSampler(
    dataset,
    problem,
    Vi_groups,
    theta_0=Vtheta0_dict,
    random_seed=problem_sampler_seed,
    class_sampler_seed=class_sampler_seed,
    class_selector_seed=class_selector_seed,
    permutation_random_seed=permutation_random_seed,
    partition_sampler_seed=partition_sampler_seed,
    p_test_partition=0.5,
    labels_attr_name=labels_attr_name
)

######################################################
#### 6. evaluate the inference according to the problem [get model activations per selected layers]

fnm_suffix = (
    f'part={dataset_part}'
    f'_augnm={augmentation_set_number}'
)

activations_fnm = f'{activations_fnm_prefix}_{fnm_suffix}.hdf5'
activations_path = os.path.join(activations_dirname, activations_fnm)
    
if do_compute_activations:
    log_fnm = f'{activations_fnm_prefix}_{fnm_suffix}.log'
    log_path = os.path.join(activations_dirname, log_fnm)

    trace_fnm = f'{activations_fnm_prefix}_params_{fnm_suffix}.npz'
    trace_sampled_params_path = os.path.join(activations_dirname, trace_fnm)

    os.makedirs(activations_dirname, exist_ok=True)
    actfnm_isin_dir = activations_fnm in os.listdir(activations_dirname)
    if actfnm_isin_dir and recompute_activations:
        os.remove(activations_path)

    if (not actfnm_isin_dir) or (actfnm_isin_dir and recompute_activations):
        activations_path, variables_sigmas = sensitivity_analysis.shapley.compute_activations_wrt_shapley_effect_sampling(
            sampler,
            network,
            filtered_network_leaves,
            transform_functions_list,
            inference_params,
            dataset,
            batch_size=batch_size_activations_computing,
            Npermutations=Npermutations,
            Nouter_samples=Nouter_samples,
            Ninner_samples=Ninner_samples,
            #use_switch_transforms_indices,
            #save_output_activations=True,
            activations_path=activations_path,
            log_path=log_path,
            pre_processing_functions=pre_processing_functions,
            post_processing_functions=post_processing_functions,
            #use_class_variable=True,
            #use_permutation_variable=True,
            #use_partition_variable=False,
            #class_selector_seed=0,
            #class_sampler_seed=0,
            #permutation_random_seed=0,
            #partition_sampler_seed=0,
            variables_permutation_random_seed=variables_permutation_random_seed,
            #p_test_partition=None,
            verbose=True,
            device=device,
            trace_sampled_params_path=trace_sampled_params_path
            #labels_attr_name='labels'
        )

######################################################
#### 7. compute the Shapley values

if do_compute_values:

    network_module_names = list(filtered_network_leaves)# + ['model_outputs']

    Nvariables = len(sampler.variables)

    log_fnm = f'{values_fnm_prefix}_{fnm_suffix}.log'
    log_path = os.path.join(activations_dirname, log_fnm)

    shpv_fnm = f'{values_fnm_prefix}_{fnm_suffix}.hdf5'
    shpv_path = os.path.join(activations_dirname, shpv_fnm)

    shpv_path = sensitivity_analysis.shapley.analyze_transform_with_shapley(
        #network,
        network_module_names,
        activations_path,
        #variables_sigmas,
        Nvariables,
        Nouter_samples,
        Ninner_samples,
        log_path=log_path,
        shpv_path=shpv_path,
        Njobs=Njobs_shpv_computing,
        buffer_size=buffer_size_shpv_computing,
        class_variable=True,
        verbose=True,
        rewrite_hdf5=True,
        tmp_save_path=f'../results/tmp/1.5/{augmentation_set_number}/',
    )

    print(f'Please find the computed results in file: {shpv_path}')

    if remove_activations_hdf5:
        os.remove(activations_path)
        print(f'HDF5 file containing the activations was removed: {activations_path}')
    else:
        print(f'HDF5 file containing the activations is here: {activations_path}')


print('well done!')

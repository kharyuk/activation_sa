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
    '--values_buffer_size',
    type=int,
    default=1000,
    help='specifies the buffer size of activations per process while compiting the Sobol indices'
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
    default=0,
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
    default='cs_alexnet_imagenet_ILSVRC_activations',
    help='name prefix for hdf5 file containing the activations'
)
arg_parser.add_argument(
    '--values_fnm_prefix',
    type=str,
    default='cs_alexnet_imagenet_ILSVRC_values',
    help='name prefix for hdf5 file containing the computed values'
)

arg_parser.add_argument(
    '--Nsamples',
    type=int,
    default=2**14,
    help='Number of original samples from a dataset'
)
arg_parser.add_argument(
    '--Ninner_samples',
    type=int,
    default=20,
    help='Number of samples per parametric augmentation'
)
arg_parser.add_argument(
    '--Njobs_cs_computing',
    type=int,
    default=2,
    help='Number of parallel processes used to compute Sobol indices'        
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
    '--class_selector_seed',
    type=int,
    default=0,
    help='random state for class selector (selects a class, than class sampler sample the representative'
)
arg_parser.add_argument(
    '--augpar_sampler_seeds',
    type=str,
    default='',
    help='random states for sampling augmentation parameters'
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

# now, continue with the others
import exp_assistance


samples_per_class_train = args.samples_per_class_train
samples_per_class_valid = args.samples_per_class_valid
subset_random_state_train = args.subset_random_state_train
subset_random_state_valid = args.subset_random_state_valid

augmentation_set_number = args.augmentation_set_number
dataset_part = args.dataset_part #train_val_test
assert dataset_part in ['train', 'valid', 'test']

activations_dirname = args.activations_dirname
activations_fnm_prefix = args.activations_fnm_prefix
values_fnm_prefix = args.values_fnm_prefix

Nsamples = args.Nsamples
Ninner_samples = args.Ninner_samples
class_selector_seed = args.class_selector_seed
class_sampler_seed = args.class_sampler_seed
augpar_sampler_seeds = args.augpar_sampler_seeds
    

batch_size_activations_computing = args.batch_size_activations_computing

values_buffer_size = args.values_buffer_size
Njobs_cs_computing = args.Njobs_cs_computing

#os_environment_config = None
#if (args.os_environment_config is not None) and (len(args.os_environment_config) > 0):
#    os_environment_config = eval(args.os_environment_config)
data_dirname = args.data_dirname
device = args.device
model_dirname = args.model_dirname

remove_activations_hdf5 = bool(args.remove_activations_hdf5)

desired_image_width = args.desired_image_width
desired_image_height = args.desired_image_height

network_name = args.network_name
network_modules = eval(args.network_modules)
classification_layer_name = args.classification_layer_name



#######################################################################################################
### 1. loading all necessary packages

import time

#torch.set_num_threads(6)
import torch.nn as nn
import torchvision

import sklearn.metrics

import h5py

#import torch_learning
import torch_utils
import model_dissection

import models.pretrained_torch
import data_loader.general
import data_loader.imagenet


import sensitivity_analysis.contrast
import sensitivity_analysis.build_problem_cont
#import sensitivity_analysis.visualize

import image_transforms.pre_post_processing

######################################################
#### 2. loading ILSVRC dataset

#dataset_loader_kwargs_dict = {} # no options
#dataset = data_loader.imagenet.CustomImageNet('../data/imagenet/', 'train')

batch_size = None
num_workers = 0
labels_attr_name = 'targets'
image_shape = (3, desired_image_height, desired_image_width)


if dataset_part == 'train':
    (dataset, ), num_classes = data_loader.general.load_single_dataset_part(
        data_loader.imagenet.CustomImageNet,
        {},
        data_dirname,
        batch_size,
        num_workers=0,
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
elif dataset_part == 'valid':
    (dataset, ), num_classes = data_loader.general.load_single_dataset_part(
        data_loader.imagenet.CustomImageNet,
        {},
        data_dirname,
        batch_size,
        num_workers=0,
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
elif dataset_part == 'test':
    raise NotImplementedError
    #(dataset, ), _, num_classes = data_loader.general.load_test(
    #    data_loader.imagenet.CustomImageNet, dataset_loader_kwargs_dict, data_dirname, batch_size,
    #    num_workers, augmentation_set_test=None, subset_size=subset_size, shuffle=False, 
    #    splitting_random_state=subset_random_state, return_torch_dataset=True,
    #    return_torch_loader=False, return_plain_data=False, labels_attr_name=labels_attr_name
    #)
else:
    raise ValueError
    
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
    network, softmax_hook, classification_layer_name #['classifier.6'] # classifier.6 is the last Linear layer without non-linearity
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
    problem, transform_functions_list
) = sensitivity_analysis.build_problem_cont.build_augmented_classification_problem(
    augmentation_set_number,
    image_shape,
    num_classes
)

num_aug_params = len(problem['names'])
if len(augpar_sampler_seeds) > 0:
    augpar_sampler_seeds = eval(augpar_sampler_seeds)
    if not isinstance(augpar_sampler_seeds, int):
        assert len(augpar_sampler_seeds) == num_aug_params
else:
    augpar_sampler_seeds = np.arange(num_aug_params)

sampler = sensitivity_analysis.contrast.CustomAugParamSampler(
    problem,
    dataset,
    augpar_sampler_seeds,
    class_sampler_seed=class_sampler_seed,
    class_selector_seed=class_selector_seed,
    labels_attr_name=labels_attr_name
)

######################################################
#### 6. evaluate the inference according to the problem [get model activations per selected layers]

fnm_suffix = (
    f'part={dataset_part}'
)

activations_basename = f'{activations_fnm_prefix}_{fnm_suffix}'
activations_paths = {}
activations_fnm = f'{activations_basename}_original.hdf5'
activations_path = os.path.join(activations_dirname, activations_fnm)
activations_paths['original'] = activations_path
for aug_name in problem['aug_names']:
    activations_fnm = f'{activations_basename}_{aug_name}.hdf5'
    activations_path = os.path.join(activations_dirname, activations_fnm)
    activations_paths[aug_name] = activations_path

#print(activations_paths)

if do_compute_activations:
    log_fnm = f'{activations_fnm_prefix}_{fnm_suffix}.log'
    log_path = os.path.join(activations_dirname, log_fnm)

    trace_fnm = f'{activations_fnm_prefix}_params_{fnm_suffix}.npz'
    trace_sampled_params_path = os.path.join(activations_dirname, trace_fnm)

    os.makedirs(activations_dirname, exist_ok=True)
    already_exist_fnms = os.listdir(activations_dirname)
    actfnm_isin_dir = False
    for key in activations_paths:
        _, activations_fnm = os.path.split(activations_paths[key])
        actfnm_isin_dir = actfnm_isin_dir or (activations_fnm in already_exist_fnms)
        if actfnm_isin_dir:
            break
            
    if actfnm_isin_dir and recompute_activations:
        for _, act_path in activations_paths.items():
            if os.path.isfile(act_path):
                os.remove(act_path)

    if (not actfnm_isin_dir) or (actfnm_isin_dir and recompute_activations):
        # filtered_network_leaves without the 'model_outputs', save_output_activations controls it here
        activations_paths = sensitivity_analysis.contrast.compute_activations_wrt_contrast_stats(
            sampler,
            network,
            filtered_network_leaves,
            transform_functions_list,
            inference_params,
            dataset,
            batch_size=batch_size_activations_computing,
            Nsamples=Nsamples,
            Ninner_samples=Ninner_samples,
            #save_output_activations=True,
            activations_dirname=activations_dirname,
            activations_basename=activations_basename,
            log_path=log_path,
            pre_processing_functions=pre_processing_functions,
            post_processing_functions=post_processing_functions,
            verbose=True,
            device=device,
            trace_sampled_params_path=trace_sampled_params_path
        )
    
#corrected_pvalues_dict = multiple_tests_corection_fdr_by(pvalues_dict, alpha_fdr=0.05)


######################################################
#### 7. compute the contrast statistics [Wilcoxon signed rank test pvals, stats, pbscc, rbscc]

#value_names = ['si+sT', 'si2']

if do_compute_values:
    network_module_names = list(filtered_network_leaves) #+ ['model_outputs']

    activations_original_path = activations_paths['original']
    cs_paths = {}
    for i_aug, aug_name in enumerate(problem['aug_names']):
        print(aug_name)
        log_fnm = f'{values_fnm_prefix}_{aug_name}_{fnm_suffix}.log'
        log_path = os.path.join(activations_dirname, log_fnm)

        cs_fnm = f'{values_fnm_prefix}_{aug_name}_{fnm_suffix}.hdf5'
        cs_path = os.path.join(activations_dirname, cs_fnm)

        activations_augmented_path = activations_paths[aug_name]

        current_cs_path = sensitivity_analysis.contrast.compute_paired_wilcoxon(
            network_module_names,
            activations_original_path,
            activations_augmented_path,
            Ninner_samples=Ninner_samples,
            log_path=log_path,
            cs_path=cs_path,
            Njobs=Njobs_cs_computing,
            buffer_size=values_buffer_size,
            verbose=True,
            rewrite_hdf5=True,
            tmp_save_path=f'../results/tmp/1.1/{dataset_part}/',
        )
        print(f"{i_aug+1}/{len(problem['aug_names'])}: Please find the computed results within file: {current_cs_path}")

        if remove_activations_hdf5:
            os.remove(activations_augmented_path)
            print(f'HDF5 file containing the activations was removed: {activations_augmented_path}')
        else:
            print(f'HDF5 file containing the activations is here: {activations_augmented_path}')
        cs_paths[aug_name] = current_cs_path

    if False and remove_activations_hdf5:
        os.remove(activations_original_path)
        print(f'HDF5 file containing the activations was removed: {activations_original_path}')
    else:
        print(f'HDF5 file containing the activations is here: {activations_original_path}')
print('well done!')
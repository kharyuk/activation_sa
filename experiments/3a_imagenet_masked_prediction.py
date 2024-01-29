#### 0. parsing the input arguments ##########################################################
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--recompute_predictions',
    type=int,
    default=0
)
arg_parser.add_argument(
    '--batch_size_computing',
    type=int,
    default=100,
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
    '--Nsamples',
    type=int,
    default=50000,
    help='Number of original samples from a dataset'
)
arg_parser.add_argument(
    '--Ninner_samples',
    type=int,
    default=3,
    help='Number of samples per parametric augmentation'
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
arg_parser.add_argument(
    '--use_permutation_variable',
    type=int,
    default=0
)
arg_parser.add_argument(
    '--use_class_variable',
    type=int,
    default=0
)
arg_parser.add_argument(
    '--use_partition_variable',
    type=int,
    default=0
)
arg_parser.add_argument(
    '--sensitivity_values_name',
    type=str,
)
arg_parser.add_argument(
    '--alphas',
    type=str,
    default='[0, 0.5, 1.5]'
)
arg_parser.add_argument(
    '--percentiles',
    type=str,
    default='[0.5, 0.6, 0.7, 0.8, 0.9]'
)
arg_parser.add_argument(
    '--top_n_predictions',
    type=int,
    default=5,
)
arg_parser.add_argument(
    '--sensitivity_values_dirname',
    type=str,
    help='path to directory containing the computed sensitivity values'
)
arg_parser.add_argument(
    '--values_fnm_base',
    type=str
)
arg_parser.add_argument(
    '--output_filename_suffix',
    type=str,
    default='pred',
)
arg_parser.add_argument(
    '--results_dirname_path',
    type=str,
)

        
# make the src directory visible
import sys
sys.path.append('../src/')        
import exp_assistance

args = arg_parser.parse_args()

torch_seed = args.torch_seed
numpy_seed = args.numpy_seed
mkl_num_threads = args.mkl_num_threads
libstdcpp_path = args.libstdcpp_path
os_environment_config = None
if (args.os_environment_config is not None) and (len(args.os_environment_config) > 0):
    os_environment_config = exp_assistance.convert_argstr2dict(args.os_environment_config)

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
import sensitivity_analysis.augmentation_setting
import prediction.compute
import prediction.compute_separated
import preparation.single_unit

samples_per_class_train = args.samples_per_class_train
samples_per_class_valid = args.samples_per_class_valid
subset_random_state_train = args.subset_random_state_train
subset_random_state_valid = args.subset_random_state_valid

augmentation_set_number = args.augmentation_set_number
assert augmentation_set_number in (1, 2, 3)
dataset_part = args.dataset_part #train_val_test
assert dataset_part in ['train', 'valid', 'test']

Nsamples = args.Nsamples
Ninner_samples = args.Ninner_samples
class_selector_seed = args.class_selector_seed
class_sampler_seed = args.class_sampler_seed
augpar_sampler_seeds = args.augpar_sampler_seeds
    
batch_size_computing = args.batch_size_computing

data_dirname = args.data_dirname
device = args.device
model_dirname = args.model_dirname

desired_image_width = args.desired_image_width
desired_image_height = args.desired_image_height

network_name = args.network_name
network_modules = eval(args.network_modules)
classification_layer_name = args.classification_layer_name

recompute_predictions = bool(args.recompute_predictions)
use_permutation_variable = bool(args.use_permutation_variable)
use_class_variable = bool(args.use_class_variable)
use_partition_variable = bool(args.use_partition_variable)

assert use_permutation_variable == use_class_variable == use_partition_variable
extract_auxilliary_names = use_permutation_variable and use_class_variable and use_partition_variable

sensitivity_values_name = args.sensitivity_values_name # 'shpv'

alphas = np.array( eval(args.alphas) )
percentiles = np.array( eval(args.percentiles) )
top_n_predictions = args.top_n_predictions

sensitivity_values_dirname = args.sensitivity_values_dirname

output_filename_suffix = args.output_filename_suffix
values_fnm_base = args.values_fnm_base

results_dirname_path = args.results_dirname_path


if augmentation_set_number == 3:
    augmentation_set_numbers_list = [1, 2]
else:
    augmentation_set_numbers_list = [augmentation_set_number]


N_aux = (
    int(use_permutation_variable)
    + int(use_class_variable)
    + int(use_partition_variable)
)


predictions_fnm_base = f'{network_name}_{values_fnm_base}_{output_filename_suffix}'

shpv_group_indices_dict = {}
for aug_set_num in augmentation_set_numbers_list:
    shpv_group_indices_dict[
        aug_set_num
    ] = sensitivity_analysis.augmentation_setting.get_group_variables_indices(
        aug_set_num,
        use_permutation_variable=use_permutation_variable,
        use_class_variable=use_class_variable,
        use_partition_variable=use_partition_variable,
    )

values_fnms_dict = preparation.single_unit.extract_massive_values_fnms(
    network_name,
    values_fnm_base,
    augmentation_set_numbers_list,
    prefix=None
);


loaded_sensitivity_values_dict = prediction.compute.get_all_values(
    values_fnms_dict,
    sensitivity_values_dirname,
    network_modules,
    sensitivity_values_name,
    augmentation_set_numbers_list,
    shpv_group_indices_dict,
    extract_auxilliary_names=extract_auxilliary_names,
)
loaded_sensitivity_values_dict = dict(
    (key, torch.from_numpy(val)) for key, val in loaded_sensitivity_values_dict.items()
)


############################################



#######################################################################################################
### 1. loading all necessary packages

import time

#import torch_learning
import torch_utils

import models.pretrained_torch
import data_loader.general
import data_loader.imagenet

import sensitivity_analysis.contrast
import sensitivity_analysis.build_problem_cont

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
print(augmentation_set_numbers_list, num_aug_params, sampler.num_augs)

######################################################
#### 6. evaluate the inference according to the problem [get model activations per selected layers]

fnm_suffix = (
    f'part={dataset_part}'
)

log_fnm_base = f'{sensitivity_values_name}_{predictions_fnm_base}_{fnm_suffix}'
results_fnm_base = f'{sensitivity_values_name}_{predictions_fnm_base}_{fnm_suffix}'
print(results_fnm_base)
print(device)

os.makedirs(results_dirname_path, exist_ok=True)
predictions_path = prediction.compute_separated.compute_predictions_wrt_single_augs(
    sampler,
    network,
    filtered_network_leaves,
    transform_functions_list,
    inference_params,
    dataset,
    batch_size=batch_size_computing,
    Nsamples=Nsamples,
    Ninner_samples=Ninner_samples,
    N_aux=N_aux,
    augmentation_set_numbers_list=augmentation_set_numbers_list,
    sensitivity_values_dict=loaded_sensitivity_values_dict,
    alphas=alphas,
    percentiles=percentiles,
    save_dirname=results_dirname_path,
    save_filename_base=results_fnm_base,
    top_n=top_n_predictions,
    #save_basename='modified_activations_pred',
    #log_path=log_path,#'modified_activations_pred.log',
    log_filename_base=log_fnm_base,
    pre_processing_functions=pre_processing_functions,
    post_processing_functions=post_processing_functions,
    verbose=True,
    device=device,
    no_aug_key = 'original',
    y_true_key = 'true_labels',
)

results_path = prediction.compute_separated.gather_results(
    results_dirname_path=results_dirname_path,
    results_filename_base=results_fnm_base,
    results_dirname_path2=sensitivity_values_dirname,
    y_true_key='true_labels',
    label_name='labels',
    predictions_name='predictions',
    probas_name='probas',
)

print(f'HDF5 file containing the results is here: {results_path}')
print('well done!')

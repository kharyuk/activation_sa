import functools
import collections
import json

import torch
import torch.nn as nn
import numpy as np
import h5py
import torchvision

import sensitivity_analysis.augmentation_setting

_HSVG_channel_names = [
    'hue',
    'saturation',
    'value',
    #, 'grayscale'
]

# https://stackoverflow.com/questions/65831101/is-there-a-split-equivalent-to-torch-nn-sequential
class SimultaneousInferenceSingleChannelModule(nn.Module):
    def __init__(self, modules_dict, in_channels_list):
        super(SimultaneousInferenceSingleChannelModule, self).__init__()
        assert len(modules_dict) == len(in_channels_list)
        self.modules_dict = nn.ModuleDict(modules_dict)
        self.in_channels_list = in_channels_list
    
    def forward(self, input_batch, no_return=True):
        input_batch = extract_HSVG_channels(input_batch, stack_result=False)
        input_batch = torch.cat(input_batch, dim=0)
        #print(input_batch.shape)
        result = {}
        for i, name in enumerate(self.modules_dict):
            output = multiplex_single_channel(input_batch, self.in_channels_list[i])
            #print(output.shape, name)
            output = self.modules_dict[name](output)
            #print(output.shape, name)
            if not no_return:
                result[name] = output
        if not no_return:
            return result
        #return collections.OrderedDict(
        #    (name, module(input_batch)) for name, module in self.modules.items()
        #)

class SelectSubsetOutput(nn.Module):
    def __init__(self, module, indices):
        super(SelectSubsetOutput, self).__init__()
        self.module = module
        self.indices = indices
        #self.in_channels_list = indices
        
    def forward(self, input_batch):
        output_batch = self.module(input_batch)
        return output_batch[:, self.indices] #####
        #return output_batch[:, self.in_channels_list] #####
        

# https://linuxtut.com/en/20819a90872275811439/
def rgb2hsv(input_batch, eps=1e-20, normalize_H=True, stack_result=False):
    shape = input_batch.shape
    assert len(shape) == 4 # (samples, channels, height, width) 
    assert input_batch.shape[1] == 3
    
    red, green, blue = input_batch[:, 0], input_batch[:, 1], input_batch[:, 2]
    min_value, min_ind = input_batch.min(dim=1)
    max_value, max_ind = input_batch.max(dim=1)
    dval = max_value - min_value + eps

    h1 = 60.*(1 + (green - red)/dval)
    h2 = 60.*(3 + (blue - green)/dval)
    h3 = 60.*(5 + (red - blue)/dval)

    H = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=min_ind.unsqueeze(0)).squeeze(0)
    if normalize_H:
        H = H / 360.
    S = dval / (max_value + eps)
    H = H[:, None]
    S = S[:, None]
    max_value = max_value[:, None]
    if stack_result:
        return torch.stack((H, S, max_value), dim=1)
    return  H, S, max_value

def multiplex_single_channel(input_batch, c_out):
    shape = input_batch.shape
    assert len(shape) == 4
    assert shape[1] == 1

    return input_batch.repeat((1, c_out, 1, 1))

def subset_conv2d_module(conv_module, channel_indices):
    out_channels = len(channel_indices)
    duplicated_module = nn.Conv2d(
        in_channels=conv_module.in_channels,
        out_channels=out_channels,
        kernel_size=conv_module.kernel_size,
        stride=conv_module.stride,
        padding=conv_module.padding,
        dilation=conv_module.dilation,
        groups=conv_module.groups,
        bias=conv_module.bias is not None,
        padding_mode=conv_module.padding_mode
    )

    duplicated_module.weight.data = conv_module.weight.data[channel_indices, :, :, :]
    if conv_module.bias is not None:
        duplicated_module.bias.data = conv_module.bias.data[channel_indices]
    return duplicated_module

def subset_module(module, neuron_indices):
    if isinstance(module, nn.Conv2d):
        return subset_conv2d_module(module, neuron_indices)
    raise ValueError


# https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
def extract_sequential_unit(network, module_names, neuron_indices_dict):
    # TODO: residual connections ? ? ? ? ? ? ?
    modules_list = []
    for module_name in module_names:
        current_neuron_indices = neuron_indices_dict.get(module_name, None)
        current_names = module_name.split(sep='.')
        current_module = functools.reduce(getattr, current_names, network)
        if current_neuron_indices is not None:
            current_module = subset_module(current_module, current_neuron_indices)
        modules_list.append(current_module)
    return nn.Sequential(*modules_list)

def extract_sequential_unit_with_residual_subunits(network, module_names, neuron_indices_dict):
    # check it
    modules_list = []
    for module_name in module_names:
        current_neuron_indices = neuron_indices_dict.get(module_name, None)
        current_names = module_name.split(sep='.')
        current_module = functools.reduce(getattr, current_names, network)
        if current_neuron_indices is not None:
            current_module = SelectSubsetOutput(current_module, current_neuron_indices)
        modules_list.append(current_module)
    return nn.Sequential(*modules_list)

def get_units_of_interest(network, config_list):
    modules_dict = {}
    for config_dict in config_list:
        modules_names = config_dict['modules_names']
        neuron_indices_dict = config_dict['neuron_indices_dict']
        module_type = config_dict['module_type']
        unit_name = config_dict['unit_name']
        if module_type.lower() == 'sequential':
            module = extract_sequential_unit(network, modules_names, neuron_indices_dict)
        elif module_type.lower() == 'residual':
            module = extract_sequential_unit_with_residual_subunits(
                network, modules_names, neuron_indices_dict
            )
        else:
            raise ValueError
        modules_dict[unit_name] = module
    return modules_dict

def extract_HSVG_channels(input_batch, stack_result=False, add_grayscale=False):
    H, S, V = rgb2hsv(input_batch, eps=1e-20, normalize_H=True, stack_result=False)
    result = [H, S, V]
    if add_grayscale:
        G = torchvision.transforms.functional.rgb_to_grayscale(
            input_batch, num_output_channels=1
        )
        result.append(G)
    if stack_result:
        return torch.stack(result, dim=1)
    return result

def save_single_channel_activations(
    activations_path, unit_name, module, inputs, activations
):
    n_featured_chans = len(_HSVG_channel_names) # global
    activations = activations.detach().cpu().numpy().astype('float32')
    shape = activations.shape[1:]
    activations = activations.reshape((n_featured_chans, -1)+shape)
    len_b = activations.shape[1]
    activations = np.reshape(activations, (n_featured_chans, len_b, -1))
    with h5py.File(activations_path, 'a') as activ:
        if unit_name in activ:
            current_group = activ[unit_name]
            for i, input_channel_name in enumerate(_HSVG_channel_names):
                len_a = current_group[input_channel_name].shape[0]
                current_group[input_channel_name].resize(len_a+len_b, axis=0)
                current_group[input_channel_name][-len_b:] = activations[i]
        else:
            current_group = activ.create_group(unit_name)
            for j, input_channel_name in enumerate(_HSVG_channel_names):
                current_group.create_dataset(
                    name=input_channel_name,
                    data=activations[j],
                    #dtype=activations.dtype,
                    compression="gzip",
                    chunks=True,
                    maxshape=(None, activations.shape[-1])
                )
        if 'shape' in current_group.attrs:
            #assert current_group.attrs['shape'] == shape
            pass
        else:
            current_group.attrs.create('shape', data=shape)
                
def save_config_to_json(config_list, save_path):
    #for i in range(len(config_list)):
    #    for x_unit in config_list[i]['neuron_indices_dict']:
    #        if config_list[i]['neuron_indices_dict'][x_unit] = (
    #            config_list[i]['neuron_indices_dict'][x_unit].tolist()
    #        )
    with open(save_path, 'w') as f_out:
        json.dump(config_list, f_out)

def load_config_from_json(save_path):
    with open(save_path, 'r') as f_in:
        config_list = json.load(f_in)
    #for i in range(len(config_list)):
    #    for x_unit in config_list[i]['neuron_indices_dict']:
    #        config_list[i]['neuron_indices_dict'][x_unit] = torch.LongTensor(
    #            config_list[i]['neuron_indices_dict'][x_unit]
    #        )
    return config_list

def ind2pair(ind, ncols_full):
    i, j = ind//ncols_full+1, ind%ncols_full+1
    return i, j

def pair2ind(ij, ncols_full):
    return (ij[0]-1)*ncols_full + (ij[1]-1)

def extract_neuron_indices(config_list):
    for i, local_config in enumerate(config_list):
        config_list[i]['neuron_indices_dict'] = config_list[i].get('neuron_indices_dict', {})
        for module_name in local_config['neurons']:
            #known_indices = config_list[i]['neuron_indices_dict'].get(
            #    module_name, []
            #)
            #new_indices = list(local_config['neurons'][module_name].keys())
            indices = list(map(int, local_config['neurons'][module_name].keys()))
            config_list[i]['neuron_indices_dict'][module_name] = torch.LongTensor(
                indices
            )
    return config_list

def extract_augmentation_names_dict(
    augmentation_set_numbers_list,
    extract_auxilliary_names=False
):
    augmentation_names_dict = {}
    for augmentation_set_number in augmentation_set_numbers_list:
        augmentation_names_dict[augmentation_set_number] = (
            sensitivity_analysis.augmentation_setting.get_augmentation_transforms_names(
                augmentation_set_number,
                use_permutation_variable=extract_auxilliary_names,
                use_class_variable=extract_auxilliary_names,
                use_partition_variable=extract_auxilliary_names,
            ) # get only augmentation names, without additional variables
        )
        augmentation_names_dict[augmentation_set_number] = list(
            map(
                lambda x: x.replace('_transform', ''),
                augmentation_names_dict[augmentation_set_number]
            )
        )
    return augmentation_names_dict

def extract_massive_values_fnms(
    network_name,
    values_fnm_base,
    augmentation_set_numbers_list,
    prefix=None
):
    values_fnms_dict = {}
    values_names = ['cs', 'sitv', 'si', 'shptv', 'shpv']
    #augmentation_set_numbers_list = [1, 2]

    #values_fnm_base = 'imagenet_ILSVRC_values'
    augmentation_names_dict = extract_augmentation_names_dict(augmentation_set_numbers_list)
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )

    dataset_parts = ['train', 'valid']
    dataset_part = '+'.join(dataset_parts)
    for current_valname in values_names[1:]:
        values_fnms_dict[current_valname] = {}
        for augmentation_set_number in augmentation_set_numbers_list:
            fnm_suffix = (
                f'part={dataset_part}'
                f'_augnm={augmentation_set_number}'
            )
            current_fnm = f'{current_valname}_{network_name}_{values_fnm_base}_{fnm_suffix}.hdf5'
            if prefix is not None:
                current_fnm = f'{prefix}_{current_fnm}'
            values_fnms_dict[current_valname][augmentation_set_number] = current_fnm

    current_valname = 'cs' # special case
    values_fnms_dict[current_valname] = {}
    for dataset_part in dataset_parts:
        values_fnms_dict[current_valname][dataset_part] = []
        for i_aug, aug_name in enumerate(augmentation_names):
            fnm_suffix = (
                f'{aug_name}'
                f'_part={dataset_part}'
                f'_fdr-by-corrected.'
            )
            current_fnm = f'{current_valname}_{network_name}_{values_fnm_base}_{fnm_suffix}.hdf5'
            if prefix is not None:
                current_fnm = f'{prefix}_{current_fnm}'
            values_fnms_dict[current_valname][dataset_part].append(current_fnm)
    return values_fnms_dict

def extract_channeled_single_units_values_fnms(
    network_name,
    values_fnm_base,
    augmentation_set_numbers_list,
    channels
):
    # better to write something more clear than efficient
    values_fnms_dict = {}
    for channel_name in channels:
        values_fnms_dict[channel_name] = extract_massive_values_fnms(
            network_name,
            values_fnm_base,
            augmentation_set_numbers_list,
            prefix=channel_name
        )

    return values_fnms_dict

def get_network_module_name(neuron_config_dict):
    return f"modules_dict.{neuron_config_dict['unit_name']}.{len(neuron_config_dict['modules_names'])-1}"

def get_neuron_indices_list(neuron_config_dict, key=None):
    if key is None:
        key = list(neuron_config_dict['neuron_indices_dict'].keys())[0]
    return neuron_config_dict['neuron_indices_dict'][key].tolist()


if __name__ == '__main__':
    # move to jupy-jupy
    config_list = []
    config_dict_sample = {
        'modules_names': ['net.u1.conv', 'net.u1.relu'],
        'neuron_indices_dict': {'net.u1.conv': torch.LongTensor([0, 3, 7])},
        'module_type': 'sequential',
        'unit_name': 'unit1'
    }
    config_list.append(config_dict_sample)
    save_config_to_json(config_list, save_path)
    
    '''
    units = get_units_of_interest(network, config_list)
    #common_inference_module = SimultaneousInferenceModule(units)

    HSVG_batch_list = extract_HSVG_channels(
        input_batch, stack_result=False
    )

    for i, sc_batch in enumerate(HSVG_batch_list):
        input_channel_name = HSVG_channel_names[i]
        #output_batch_list = common_inference_module(sc_batch)
        for j, cofig_dict in enumerate(config_list):
            unit_name = cofig_dict['unit_name']
            activations = units[j](sc_batch)
            save_single_channel_activations(
                activations_path, unit_name, input_channel_name, activations
            )


        # save to the relevant component file
        # ... tired. and watched another rude/awful/nasty male/female species around today
        # Wish they are already gone. Want to be isolated from all this s**t.
        
    '''

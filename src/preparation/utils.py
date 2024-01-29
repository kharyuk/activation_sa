import functools

import numpy as np

from . import single_unit
from . import visualize

def get_original_and_channeled_values(
    hsv_values_fnms_dict,
    su_activations_dirname,
    values_fnms_dict,
    activations_dirname,
    analysis_config_list,
    network_modules,
    values_names_dict,
    augmentation_set_numbers_list,
    shpv_group_indices_dict,
    extract_auxilliary_names=True,
):
    results = []
    results_orig = []
    
    row_chans_names = list(hsv_values_fnms_dict.keys())
    assert set(row_chans_names).issubset(set(single_unit._HSVG_channel_names))
    n_chans = len(row_chans_names)
    row_vals_names = functools.reduce(
        lambda x, y: x+y, values_names_dict.values()
    )
    n_vals = len(row_vals_names)
    max_len_row_vals = max(map(len, row_vals_names))
    
    augmentation_and_auxilliary_names_dict = single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_and_auxilliary_names = functools.reduce(
        lambda x, y: x+y, augmentation_and_auxilliary_names_dict.values()
    )
    n_aug_groups = len(augmentation_set_numbers_list)
    n_aug_aux = len(augmentation_and_auxilliary_names)

    for i_mn, neurons_config_dict in enumerate(analysis_config_list):

        ijs = list(neurons_config_dict['neuron_indices_dict'].values())[0]
        ijs = np.array(ijs.tolist())

        neuron_indices_list = single_unit.get_neuron_indices_list(neurons_config_dict)

        module_name = single_unit.get_network_module_name(neurons_config_dict)

        dataset_part = None
        current_result = {}
        current_results_orig = {}
        for augmentation_set_number in augmentation_set_numbers_list:
            current_chan_results = []
            current_chan_results_orig = []
            for i_block_row, block_row_name in enumerate(row_chans_names): # select channel, i.e., hue, value, etc
                current_values_fnm_dict = hsv_values_fnms_dict[block_row_name]
                i_row = i_block_row*n_vals
                current_chan_val_results = []
                for i_valkey, value_key in enumerate(values_names_dict):
                    for values_name in values_names_dict[value_key]: # select concrete measurement, i.e., si, etc
                        c_values_name = visualize._values_names_dict[values_name]                            
                        local_values = visualize.get_conv2d_unit_values(
                            current_values_fnm_dict,
                            su_activations_dirname,
                            module_name,
                            value_key,
                            values_name=c_values_name,
                            augmentation_set_number=augmentation_set_number,
                            dataset_part=dataset_part,
                            slice_num=None,
                            values_func=None if values_name not in visualize._values_funcs else lambda x: (
                                visualize._values_funcs[values_name](
                                    x,
                                    shpv_group_indices_dict[augmentation_set_number]
                                )
                            ),
                            shpv_normalize=True,
                        )
                        current_chan_val_results.append(local_values)
                        del local_values;
                        if i_block_row == 0:
                            local_values = visualize.get_conv2d_unit_values(
                                values_fnms_dict,
                                activations_dirname,
                                network_modules[i_mn],
                                value_key,
                                values_name=c_values_name,
                                augmentation_set_number=augmentation_set_number,
                                dataset_part=dataset_part,
                                slice_num=None,
                                values_func=None if values_name not in visualize._values_funcs else lambda x: (
                                    visualize._values_funcs[values_name](
                                        x,
                                        shpv_group_indices_dict[augmentation_set_number]
                                    )
                                ),
                                shpv_normalize=True,
                            )
                            current_chan_results_orig.append(local_values[:, ijs, :, :])
                            del local_values;
                current_chan_results.append(current_chan_val_results)
                del current_chan_val_results;
            current_result[augmentation_set_number] = np.array(current_chan_results)
            current_results_orig[augmentation_set_number] = np.array(current_chan_results_orig)
            del current_chan_results, current_chan_results_orig;
            #print(current_results_orig[augmentation_set_number].shape)
        results.append(current_result)
        results_orig.append(current_results_orig)
        del current_result, current_results_orig;
    return results, results_orig

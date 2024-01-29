import warnings

import numpy as np
import scipy.stats

import prediction.utils
from . import compute

_nan_policy_list = ('all', 'any')

def extract_unit_mean_pixelwise_correlation_patterns(
    values_results_hsv,
    values_results_orig,
    network_modules,
    augmentation_set_number,
    chans_name_list,
    value_name_list,
    extract_auxilliary_names=True,
    nan_mask_policy='any',
):
    assert nan_mask_policy in _nan_policy_list
    if nan_mask_policy == 'any':
        nan_mask_func = np.any
    elif nan_mask_policy == 'all':
        nan_mask_func = np.all
        
    #c_augaux_names = augmentation_and_auxilliary_names_dict[augset_num]
    _, c_augaux_names =  (
        prediction.utils.get_shortened_variable_names_single_augset(
            augmentation_set_number,
            extract_auxilliary_names=extract_auxilliary_names,
        )
    )
    n_vars = len(c_augaux_names)
    mask = np.ones((n_vars-1, n_vars-1))
    mask_ind = np.tril_indices(n_vars-1, 0, m=n_vars-1)
    mask[mask_ind] = 0
    n_layers = len(values_results_hsv)
    assert n_layers == len(values_results_orig)
    n_input_chans = len(chans_name_list)
    n_vals = len(value_name_list)
    
    mean_cormat_hsv = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    std_cormat_hsv = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    nanmask_cormat_hsv = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars),
        dtype='i',
    )
    
    mean_cormat_orig = np.empty(
        (n_layers, n_vals, n_vars, n_vars)
    )
    std_cormat_orig = np.empty(
        (n_layers, n_vals, n_vars, n_vars)
    )
    nanmask_cormat_orig = np.empty(
        (n_layers, n_vals, n_vars, n_vars),
        dtype='i',
    )
    
    mean_cormat_diff = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    std_cormat_diff = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    nanmask_cormat_diff = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars),
        dtype='i',
    )
    
    for i_layer in range(n_layers):
        tmp = values_results_hsv[i_layer][augmentation_set_number]
        tmp2 = values_results_orig[i_layer][augmentation_set_number]
        #print(np.isnan(tmp).any(), np.isnan(tmp2).any())
        #tmp[np.isnan(tmp)] = 0.
        #tmp2[np.isnan(tmp2)] = 0.
        c_n_chan_input, c_n_vals, c_n_vars, c_n_units, _, _ = tmp.shape
        tmp = tmp.reshape(
            (c_n_chan_input, c_n_vals, c_n_vars, c_n_units, -1)
        )
        tmp2 = tmp2.reshape(
            (c_n_vals, c_n_vars, c_n_units, -1)
        )
        c_n_pix = tmp.shape[-1]
        for i_val, val_name in enumerate(value_name_list):
            for i_ch, chan_name in enumerate(chans_name_list):
                cor_mat = []
                cor_mat_nan = []
                if i_ch == 0:
                    cor_mat2 = []
                    cor_mat2_nan = []
                for i_unit in range(c_n_units):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cur_cor, cur_cor_nan = compute.custom_spearmanr(
                            tmp[i_ch, i_val, :, i_unit, :],
                            axis=1,
                        )
                        cor_mat.append(cur_cor)
                        cor_mat_nan.append(cur_cor_nan)
                        if i_ch == 0:
                            cur_cor2, cur_cor2_nan = compute.custom_spearmanr(
                                tmp2[i_val, :, i_unit, :],
                                axis=1,
                            )
                            cor_mat2.append(cur_cor2)
                            cor_mat2_nan.append(cur_cor2_nan)
                if i_ch == 0:
                    cor_mat2 = np.array(cor_mat2)
                    cor_mat2_nan = np.array(cor_mat2_nan)
                    mean_cormat_orig[i_layer, i_val, :, :] = np.mean(cor_mat2, axis=0)
                    std_cormat_orig[i_layer, i_val, :, :] = np.std(cor_mat2, axis=0)
                    nanmask_cormat_orig[i_layer, i_val, :, :] = nan_mask_func(cor_mat2_nan, axis=0)
                    
                cor_mat = np.array(cor_mat)
                cor_mat_nan = np.array(cor_mat_nan)
                mean_cormat_hsv[i_layer, i_ch, i_val, :, :] = np.mean(cor_mat, axis=0)
                std_cormat_hsv[i_layer, i_ch, i_val, :, :] = np.std(cor_mat, axis=0)
                nanmask_cormat_hsv[i_layer, i_ch, i_val, :, :] = nan_mask_func(cor_mat_nan, axis=0)
                
                diff_cor_mat = np.abs(cor_mat - cor_mat2)
                
                mean_cormat_diff[i_layer, i_ch, i_val, :, :] = np.mean(diff_cor_mat, axis=0)
                std_cormat_diff[i_layer, i_ch, i_val, :, :] = np.std(diff_cor_mat, axis=0, ddof=1)
                if nan_mask_policy == 'all':
                    nanmask_cormat_diff[i_layer, i_ch, i_val, :, :] = (
                        nanmask_cormat_orig[i_layer, i_val, :, :]
                        & nanmask_cormat_hsv[i_layer, i_ch, i_val, :, :]
                    )
                elif nan_mask_policy == 'any':
                    nanmask_cormat_diff[i_layer, i_ch, i_val, :, :] = (
                        nanmask_cormat_orig[i_layer, i_val, :, :]
                        | nanmask_cormat_hsv[i_layer, i_ch, i_val, :, :]
                    )
                
    return (
        (mean_cormat_hsv, std_cormat_hsv, nanmask_cormat_hsv),
        (mean_cormat_orig, std_cormat_orig, nanmask_cormat_orig),
        (mean_cormat_diff, std_cormat_diff, nanmask_cormat_diff),
    )

def mean_pattern_func(a, b):
    tmp = np.abs(a - b)
    return np.mean(tmp), np.std(tmp)

def corr_pattern_func(a, b):
    tmp = scipy.stats.spearmanr(a, b)
    tmp = tmp[0]
    return tmp, np.isnan(tmp)
    #return tmp.statistic

def extract_hsv_correlation_layerwise_diff_patterns(
    mean_cormat_hsv,
    mean_cormat_orig,
    network_modules,
    augmentation_set_number,
    chans_name_list,
    value_name_list,
    extract_auxilliary_names=True,
):
    _, c_augaux_names =  (
        prediction.utils.get_shortened_variable_names_single_augset(
            augmentation_set_number,
            extract_auxilliary_names=extract_auxilliary_names,
        )
    )
    n_vars = len(c_augaux_names)
    
    n_layers = len(network_modules)
    n_input_chans = len(chans_name_list)
    n_vals = len(value_name_list)
    assert mean_cormat_hsv.shape == (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    assert mean_cormat_orig.shape == (n_layers, n_vals, n_vars, n_vars)
    
    ind_hsv = np.triu_indices(n_vars, k=1)
    ind_hsv_orig = np.triu_indices(n_vars)
    
    cm_mean_hsv_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_std_hsv_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_diff_nanmask = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    
    cm_mean_hsv_orig_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_std_hsv_orig_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_orig_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_orig_diff_nanmask = np.empty((n_input_chans, n_vals, n_layers, n_layers))

    for i_val, row_val_name in enumerate(value_name_list):
        for i_ch, chan_name in enumerate(chans_name_list):
            for i_layer1 in range(n_layers):
                for i_layer2 in range(i_layer1, n_layers):
                    cur_mean, cur_std = mean_pattern_func(
                        mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv[0], ind_hsv[1]],
                        mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                    )
                    cm_mean_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = cur_mean
                    cm_mean_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = cur_mean
                    cm_std_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = cur_std
                    cm_std_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = cur_std

                    cur_mean, cur_std = mean_pattern_func(
                        mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]],
                        mean_cormat_orig[i_layer2, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                    )
                    cm_mean_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = cur_mean
                    cm_std_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = cur_std

                    
                    cur_spear_corr, cur_spear_corr_nanmask = corr_pattern_func(
                        mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv[0], ind_hsv[1]],
                        mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                    )
                    cm_corr_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr
                    cm_corr_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = cur_spear_corr
                    
                    cm_corr_hsv_diff_nanmask[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr_nanmask
                    cm_corr_hsv_diff_nanmask[i_ch, i_val, i_layer2, i_layer1] = cur_spear_corr_nanmask
                    

                    cur_spear_corr, cur_spear_corr_nanmask = corr_pattern_func(
                        mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]],
                        mean_cormat_orig[i_layer2, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                    )
                    cm_corr_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr
                    cm_corr_hsv_orig_diff_nanmask[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr_nanmask
                    if i_layer1 != i_layer2:
                        cur_mean, cur_std = mean_pattern_func(
                            mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]],
                            mean_cormat_orig[i_layer1, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                        )
                        cm_mean_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = cur_mean
                        cm_std_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = cur_std
                        
                        
                        cur_spear_corr, cur_spear_corr_nanmask = corr_pattern_func(
                            mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]],
                            mean_cormat_orig[i_layer1, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                        )
                        cm_corr_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = cur_spear_corr
                        cm_corr_hsv_orig_diff_nanmask[i_ch, i_val, i_layer2, i_layer1] = cur_spear_corr_nanmask
                    
                    
                    '''
                    cm_mean_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = np.mean(
                        np.abs(
                            mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                            - mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                        )
                    )
                    cm_mean_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = cm_mean_hsv_diff[i_ch, i_val, i_layer1, i_layer2]
                    
                    cm_std_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = np.std(
                        np.abs(
                            mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                            - mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                        )
                    )
                    cm_std_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = cm_std_hsv_diff[i_ch, i_val, i_layer1, i_layer2]
                    
                    
                    cm_mean_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = np.mean(
                        np.abs(
                            mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                            - mean_cormat_orig[i_layer2, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                        )
                    )
                    cm_std_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = np.std(
                        np.abs(
                            mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                            - mean_cormat_orig[i_layer2, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                        )
                    )
                    
                    if i_layer1 != i_layer2:
                        #cm_mean_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = cm_mean_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2]
                        cm_mean_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = np.mean(
                            np.abs(
                                mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                                - mean_cormat_orig[i_layer1, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                            )
                        )
                        #cm_std_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = cm_std_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2]
                        cm_std_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = np.std(
                            np.abs(
                                mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                                - mean_cormat_orig[i_layer1, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                            )
                        )
                    ''';
                    
    return (
        (cm_mean_hsv_diff, cm_std_hsv_diff, cm_corr_hsv_diff, cm_corr_hsv_diff_nanmask),
        (cm_mean_hsv_orig_diff, cm_std_hsv_orig_diff, cm_corr_hsv_orig_diff, cm_corr_hsv_orig_diff_nanmask),
    )
        

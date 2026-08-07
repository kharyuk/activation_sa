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
    ci_alpha=0.05,
    ci_n_bootstraps=1000,
    ci_random_state=0
):
    assert nan_mask_policy in _nan_policy_list
    if nan_mask_policy == 'any':
        nan_mask_func = np.any
    elif nan_mask_policy == 'all':
        nan_mask_func = np.all
    rng = np.random.default_rng(ci_random_state)
        
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
    ci_lower_cormat_hsv = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    ci_upper_cormat_hsv = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    nanmask_cormat_hsv = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars),
        dtype='i',
    )
    
    mean_cormat_orig = np.empty(
        (n_layers, n_vals, n_vars, n_vars)
    )
    ci_lower_cormat_orig = np.empty(
        (n_layers, n_vals, n_vars, n_vars)
    )
    ci_upper_cormat_orig = np.empty(
        (n_layers, n_vals, n_vars, n_vars)
    )
    nanmask_cormat_orig = np.empty(
        (n_layers, n_vals, n_vars, n_vars),
        dtype='i',
    )
    
    mean_cormat_diff = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    ci_lower_cormat_diff = np.empty(
        (n_layers, n_input_chans, n_vals, n_vars, n_vars)
    )
    ci_upper_cormat_diff = np.empty(
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
                    tf_cor_mat2 = compute.fisher_z(cor_mat2)
                    cor_mat2_nan = np.array(cor_mat2_nan)
                    mean_cormat_orig[i_layer, i_val, :, :] = compute.inv_fisher_z(np.mean(tf_cor_mat2, axis=0))
                    #std_cormat_orig[i_layer, i_val, :, :] = np.std(cor_mat2, axis=0)
                    nanmask_cormat_orig[i_layer, i_val, :, :] = nan_mask_func(cor_mat2_nan, axis=0)
                    k_bootstrap = len(tf_cor_mat2)
                    bootstrapped_means = []
                    for i_bootstrap in range(ci_n_bootstraps):
                        c_ind_bootstrap = rng.choice(k_bootstrap, size=k_bootstrap, replace=True)
                        c_mean_bootstrap = tf_cor_mat2[c_ind_bootstrap].mean(axis=0)
                        bootstrapped_means.append(c_mean_bootstrap)
                    ci_lower = np.quantile(bootstrapped_means, q=0.5*ci_alpha, axis=0)
                    ci_upper = np.quantile(bootstrapped_means, q=1-0.5*ci_alpha, axis=0)
                    ci_lower_cormat_orig[i_layer, i_val, :, :] = compute.inv_fisher_z(ci_lower)
                    ci_upper_cormat_orig[i_layer, i_val, :, :] = compute.inv_fisher_z(ci_upper)
                    
                cor_mat = np.array(cor_mat)
                tf_cor_mat = compute.fisher_z(cor_mat)
                cor_mat_nan = np.array(cor_mat_nan)
                mean_cormat_hsv[i_layer, i_ch, i_val, :, :] = compute.inv_fisher_z(np.mean(tf_cor_mat, axis=0))
                #std_cormat_hsv[i_layer, i_ch, i_val, :, :] = np.std(cor_mat, axis=0)
                nanmask_cormat_hsv[i_layer, i_ch, i_val, :, :] = nan_mask_func(cor_mat_nan, axis=0)
                k_bootstrap = len(cor_mat)
                bootstrapped_means = []
                for i_bootstrap in range(ci_n_bootstraps):
                    c_ind_bootstrap = rng.choice(k_bootstrap, size=k_bootstrap, replace=True)
                    c_mean_bootstrap = tf_cor_mat[c_ind_bootstrap].mean(axis=0)
                    bootstrapped_means.append(c_mean_bootstrap)
                ci_lower = np.quantile(bootstrapped_means, q=0.5*ci_alpha, axis=0)
                ci_upper = np.quantile(bootstrapped_means, q=1-0.5*ci_alpha, axis=0)
                ci_lower_cormat_hsv[i_layer, i_ch, i_val, :, :] = compute.inv_fisher_z(ci_lower)
                ci_upper_cormat_hsv[i_layer, i_ch, i_val, :, :] = compute.inv_fisher_z(ci_upper)
                
                diff_cor_mat = np.abs(cor_mat - cor_mat2)
                mean_cormat_diff[i_layer, i_ch, i_val, :, :] = np.mean(diff_cor_mat, axis=0)
                #std_cormat_diff[i_layer, i_ch, i_val, :, :] = np.std(diff_cor_mat, axis=0, ddof=1)
                k_bootstrap = len(diff_cor_mat)
                bootstrapped_means = []
                for i_bootstrap in range(ci_n_bootstraps):
                    c_ind_bootstrap = rng.choice(k_bootstrap, size=k_bootstrap, replace=True)
                    c_mean_bootstrap = diff_cor_mat[c_ind_bootstrap].mean(axis=0)
                    bootstrapped_means.append(c_mean_bootstrap)
                ci_lower = np.quantile(bootstrapped_means, q=0.5*ci_alpha, axis=0)
                ci_upper = np.quantile(bootstrapped_means, q=1-0.5*ci_alpha, axis=0)
                ci_lower_cormat_diff[i_layer, i_ch, i_val, :, :] = ci_lower
                ci_upper_cormat_diff[i_layer, i_ch, i_val, :, :] = ci_upper
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
        (mean_cormat_hsv, ci_lower_cormat_hsv, ci_upper_cormat_hsv, nanmask_cormat_hsv),
        (mean_cormat_orig, ci_lower_cormat_orig, ci_upper_cormat_orig, nanmask_cormat_orig),
        (mean_cormat_diff, ci_lower_cormat_diff, ci_upper_cormat_diff, nanmask_cormat_diff),
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
    ci_alpha=0.05,
    ci_n_bootstraps=1000,
    ci_random_state=0,
):
    rng = np.random.default_rng(ci_random_state)
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
    cm_ci_lower_hsv_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_ci_upper_hsv_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_diff_nanmask = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    
    cm_mean_hsv_orig_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_ci_lower_hsv_orig_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_ci_upper_hsv_orig_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_orig_diff = np.empty((n_input_chans, n_vals, n_layers, n_layers))
    cm_corr_hsv_orig_diff_nanmask = np.empty((n_input_chans, n_vals, n_layers, n_layers))

    for i_val, row_val_name in enumerate(value_name_list):
        for i_ch, chan_name in enumerate(chans_name_list):
            for i_layer1 in range(n_layers):
                for i_layer2 in range(i_layer1, n_layers):
                    hsv_cormat1 = (#compute.fisher_z(
                        mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                    )
                    hsv_cormat2 = (#compute.fisher_z(
                        mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                    )
                    cur_vals = np.abs(hsv_cormat1 - hsv_cormat2)
                    cm_mean_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = np.mean(cur_vals)
                    cm_mean_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = (
                        cm_mean_hsv_diff[i_ch, i_val, i_layer1, i_layer2]
                    )
                    k_bootstrap = len(cur_vals)
                    bootstrapped_means = []
                    for i_bootstrap in range(ci_n_bootstraps):
                        c_ind_bootstrap = rng.choice(k_bootstrap, size=k_bootstrap, replace=True)
                        c_mean_bootstrap = cur_vals[c_ind_bootstrap].mean()
                        bootstrapped_means.append(c_mean_bootstrap)
                    ci_lower = np.quantile(bootstrapped_means, q=0.5*ci_alpha)
                    ci_upper = np.quantile(bootstrapped_means, q=1-0.5*ci_alpha)
                    cm_ci_lower_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = ci_lower
                    cm_ci_lower_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = ci_lower
                    cm_ci_upper_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = ci_upper
                    cm_ci_upper_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = ci_upper

                    cur_spear_corr, cur_spear_corr_nanmask = corr_pattern_func(
                        hsv_cormat1, hsv_cormat2
                        #mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv[0], ind_hsv[1]],
                        #mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv[0], ind_hsv[1]]
                    )
                    cm_corr_hsv_diff[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr
                    cm_corr_hsv_diff[i_ch, i_val, i_layer2, i_layer1] = cur_spear_corr
                    cm_corr_hsv_diff_nanmask[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr_nanmask
                    cm_corr_hsv_diff_nanmask[i_ch, i_val, i_layer2, i_layer1] = cur_spear_corr_nanmask

                    hsv_cormat1 = (#compute.fisher_z(
                        mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                    )
                    orig_cormat2 = (#compute.fisher_z(
                        mean_cormat_orig[i_layer2, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                    )
                    cur_vals = np.abs(hsv_cormat1 - orig_cormat2)
                    #cur_mean, cur_std = mean_pattern_func(hsv_cormat1, orig_cormat2)
                    #mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]],
                    #mean_cormat_orig[i_layer2, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                    cm_mean_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = np.mean(cur_vals)
                    #cm_std_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = cur_std
                    k_bootstrap = len(cur_vals)
                    bootstrapped_means = []
                    for i_bootstrap in range(ci_n_bootstraps):
                        c_ind_bootstrap = rng.choice(k_bootstrap, size=k_bootstrap, replace=True)
                        c_mean_bootstrap = cur_vals[c_ind_bootstrap].mean()
                        bootstrapped_means.append(c_mean_bootstrap)
                    ci_lower = np.quantile(bootstrapped_means, q=0.5*ci_alpha)
                    ci_upper = np.quantile(bootstrapped_means, q=1-0.5*ci_alpha)
                    cm_ci_lower_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = ci_lower
                    cm_ci_upper_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = ci_upper

                    cur_spear_corr, cur_spear_corr_nanmask = corr_pattern_func(
                        hsv_cormat1, orig_cormat2
                        #mean_cormat_hsv[i_layer1, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]],
                        #mean_cormat_orig[i_layer2, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                    )
                    cm_corr_hsv_orig_diff[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr
                    cm_corr_hsv_orig_diff_nanmask[i_ch, i_val, i_layer1, i_layer2] = cur_spear_corr_nanmask
                    
                    if i_layer1 != i_layer2:
                        hsv_cormat2 = (#compute.fisher_z(
                            mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                        )
                        orig_cormat1 = (#compute.fisher_z(
                            mean_cormat_orig[i_layer1, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
                        )
                        cur_vals = np.abs(hsv_cormat2 - orig_cormat1)
                        cm_mean_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = np.mean(cur_vals)
                        #cm_std_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = cur_std
                        k_bootstrap = len(cur_vals)
                        bootstrapped_means = []
                        for i_bootstrap in range(ci_n_bootstraps):
                            c_ind_bootstrap = rng.choice(k_bootstrap, size=k_bootstrap, replace=True)
                            c_mean_bootstrap = cur_vals[c_ind_bootstrap].mean()
                            bootstrapped_means.append(c_mean_bootstrap)
                        ci_lower = np.quantile(bootstrapped_means, q=0.5*ci_alpha)
                        ci_upper = np.quantile(bootstrapped_means, q=1-0.5*ci_alpha)
                        cm_ci_lower_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = ci_lower
                        cm_ci_upper_hsv_orig_diff[i_ch, i_val, i_layer2, i_layer1] = ci_upper
                        
                        
                        cur_spear_corr, cur_spear_corr_nanmask = corr_pattern_func(
                            hsv_cormat2, orig_cormat1
                            #mean_cormat_hsv[i_layer2, i_ch, i_val, ind_hsv_orig[0], ind_hsv_orig[1]],
                            #mean_cormat_orig[i_layer1, i_val, ind_hsv_orig[0], ind_hsv_orig[1]]
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
        (
            cm_mean_hsv_diff,
            cm_ci_lower_hsv_diff,
            cm_ci_upper_hsv_diff,
            cm_corr_hsv_diff,
            cm_corr_hsv_diff_nanmask
        ),
        (
            cm_mean_hsv_orig_diff,
            cm_ci_lower_hsv_orig_diff,
            cm_ci_upper_hsv_orig_diff,
            cm_corr_hsv_orig_diff,
            cm_corr_hsv_orig_diff_nanmask
        ),
    )
        

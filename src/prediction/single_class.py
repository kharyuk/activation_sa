import functools
import itertools

import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import preparation.single_unit

_custom_colors = [
    'maroon',
    'red',
    'orangered',
    'peru',
    #'tan',
    'goldenrod',

    'darkkhaki',
    'darkolivegreen',
    'darkseagreen',
    'mediumaquamarine',
    'teal',
    'steelblue',
    'darkslateblue',   
]

def plot_top_sensitivity_values_decay(
    loaded_sensitivity_values_dict_list,
    classifying_layer_name,
    sens_vals_names,
    augmentation_set_numbers_list,
    n_top_values=20,
    figsize=(10, 5),
    extract_auxilliary_names=False,
    plot_colors=_custom_colors,
    show=True,
):
    # plot descending sesitivity values 

    n_sens_vals = len(sens_vals_names)
    
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    n_aug = len(augmentation_names)
    
    colors = [
        #plt.cm.pink(np.linspace(0., 1., n_aug)),
        plot_colors,
        #plt.cm.CMRmap_r(np.linspace(0.1, 0.8, n_aug)),
        plot_colors,
    ]
    fig, ax = plt.subplots(
        len(augmentation_set_numbers_list), n_sens_vals, figsize=figsize, sharex=False, sharey=False
    )
    for i_cval in range(n_sens_vals):
        loaded_sensitivity_values_dict = loaded_sensitivity_values_dict_list[i_cval]
        i_caug_glob = 0
        aug_offset = 0
        for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
            caug_list = augmentation_names_dict[augmentation_set_number]
            n_caug = len(caug_list)
            
            cax = ax[i_aug, i_cval]
            for i_caug in range(n_caug):
                ind = loaded_sensitivity_values_dict[classifying_layer_name][i_caug+aug_offset].argsort()[::-1]
                vals = loaded_sensitivity_values_dict[classifying_layer_name][i_caug+aug_offset, ind][:n_top_values]
                cax.plot(
                    vals,
                    linestyle='-',
                    marker='o',
                    markersize=2,
                    alpha=0.95,
                    color=colors[i_aug][i_caug+aug_offset],
                    label=caug_list[i_caug],
                )
            aug_offset += n_caug
            i_caug_glob += 1
            cax.set_xticklabels([])
            if i_cval == 0:
                cax.set_ylabel('Sorted sensitivity values')
            #if i_aug == 0:
            #    cax.set_title('Sorted sensitivity values')
            if i_cval == n_sens_vals-1:
                cax.legend(
                    loc='center right',
                    bbox_to_anchor=(1.5, 0.5)
                )
            if i_aug == 0:
                cax.set_title(f'({sens_vals_names[i_cval]})')
            cax.grid(alpha=0.5)
    if show:
        plt.show()
        
def plot_top_mean_var_cvs_activations_decay(
    loaded_mean_activations_list,
    loaded_vars_activations_list,
    loaded_cvs_activations_list,
    sens_vals_names,
    augmentation_set_numbers_list,
    n_top_values=20,
    figsize=(10, 5),
    plot_colors=_custom_colors,
    show=True,
):
    # plot descending mean activations
    n_augsets = len(augmentation_set_numbers_list)
    n_sens_vals = len(sens_vals_names)
    row_values_names = ['means', 'CoVs']
    
    colors = [
        #plt.cm.pink(np.linspace(0., 1., n_aug)),
        #plot_colors,
        #plt.cm.CMRmap_r(np.linspace(0.1, 0.8, n_aug)),
        #plot_colors,
        ['maroon', 'steelblue'],
        ['maroon', 'steelblue'],
    ]
    fig, ax = plt.subplots(
        len(row_values_names), n_sens_vals, figsize=figsize, sharex=True, sharey=False
    )
    for i_cval in range(n_sens_vals):
        current_mean_activations = loaded_mean_activations_list[i_cval]
        current_vars_activations = loaded_vars_activations_list[i_cval]
        current_cvs_activations = loaded_cvs_activations_list[i_cval]
        i_caug_glob = 0
        aug_offset = 0
        
        for i_vl, current_values in enumerate(
            (
                current_mean_activations,
                current_cvs_activations,
            )
        ):
            cax = ax[i_vl, i_cval]
            for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
                cur_vals = current_values[i_aug].flatten()
                ind = cur_vals.argsort()[::-1]
                cax.semilogy(
                #cax.plot(
                    range(1, n_top_values+1),
                    cur_vals[ind[:n_top_values]],
                    linestyle='-',
                    marker='o',
                    markersize=2,
                    alpha=0.95,
                    color=colors[i_cval][i_aug],
                    label=f'augmentation set {augmentation_set_number}',
                )
                #cax.set_xticklabels([])
                if i_cval == 0:
                    cax.set_ylabel(f'Sorted activations {row_values_names[i_vl]}')
                if i_cval == n_sens_vals-1:
                    cax.legend(
                        loc='center right',
                        bbox_to_anchor=(1.5, 0.5)
                    )
                if i_vl == 0:
                    cur_std = current_vars_activations[i_aug].flatten()**0.5
                    cax.fill_between(
                        range(1, n_top_values+1),
                        #np.log10(
                        #    np.maximum(
                        cur_vals[ind[:n_top_values]] - cur_std[ind[:n_top_values]], #1e-8
                        #    )
                        #),  
                        #np.log10(cur_vals[ind[:n_top_values]] + cur_std[ind[:n_top_values]]),
                        cur_vals[ind[:n_top_values]] + cur_std[ind[:n_top_values]],
                        alpha=0.25,
                        color=colors[i_cval][i_aug],
                    )
                    cax.fill_between(
                        range(1, n_top_values+1),
                        #np.log10(
                        #    np.maximum(
                        cur_vals[ind[:n_top_values]] - 2*cur_std[ind[:n_top_values]], #1e-8
                        #    )
                        #),  
                        #np.log10(cur_vals[ind[:n_top_values]] + 2*cur_std[ind[:n_top_values]]),
                        cur_vals[ind[:n_top_values]] + 2*cur_std[ind[:n_top_values]],
                        alpha=0.1,
                        color=colors[i_cval][i_aug],
                    )
                    cax.fill_between(
                        range(1, n_top_values+1),
                        #np.log10(
                        #    np.maximum(
                        cur_vals[ind[:n_top_values]] - 3*cur_std[ind[:n_top_values]], #1e-8
                        #    )
                        #),  
                        #np.log10(cur_vals[ind[:n_top_values]] + 3*cur_std[ind[:n_top_values]]),
                        cur_vals[ind[:n_top_values]] + 3*cur_std[ind[:n_top_values]],
                        alpha=0.05,
                        color=colors[i_cval][i_aug],
                    )
                    
                    cax.set_title(f'({sens_vals_names[i_cval]})')
                cax.grid(alpha=0.5)
    if show:
        plt.show()
        

def extract_tables_top_sensitivity_values_decay(
    loaded_sensitivity_values_dict_list,
    classifying_layer_name,
    sens_vals_names,
    augmentation_set_numbers_list,
    ind2class_dict,
    n_top_values=10,
    extract_auxilliary_names=False,
):
    # print table for different augmentations
    #                          shpv //                            si
    # class id / class name / value // class id / class name / value

    n_sens_vals = len(sens_vals_names)
    
    head1_row = []
    for i in range(n_sens_vals):
        head1_row += [sens_vals_names[i]]*3
    head2_row = ['class_id', 'class_name', 'value']*2
    table_head = [head1_row, head2_row]
    
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    n_aug = len(augmentation_names)
        
    table_dict = {}
    for i_aug, aug_name in enumerate(augmentation_names):
        table = [[] for i in range(n_top_values)]
        for i_cval in range(n_sens_vals):
            loaded_sensitivity_values_dict = loaded_sensitivity_values_dict_list[i_cval]
            ind = loaded_sensitivity_values_dict[
                classifying_layer_name
            ][i_aug].argsort()[::-1]
            vals = loaded_sensitivity_values_dict[
                classifying_layer_name
            ][i_aug, ind][:n_top_values]
            for i_row in range(n_top_values):
                id_class = ind[i_row]
                cval = f'{vals[i_row]:.2f}'
                cval = cval.split('.')[-1]
                cval = f'.{cval}'
                table[i_row] += [
                    id_class,
                    ind2class_dict[id_class],
                    cval,
                ]
        table = pd.DataFrame(
            data=table,
            columns=pd.MultiIndex.from_arrays(table_head)
        )
        table_dict[aug_name] = table
    return table_dict

def extract_tables_top_mean_activations_decay(
    loaded_mean_activations_list,
    sens_vals_names,
    augmentation_set_numbers_list,
    ind2class_dict,
    n_top_values=10,
):
    # print table for different augmentations
    #                          shpv //                            si
    # class id / class name / value // class id / class name / value

    n_sens_vals = len(sens_vals_names)
    
    head1_row = []
    for i in range(n_sens_vals):
        head1_row += [sens_vals_names[i]]*3
    head2_row = ['class_id', 'class_name', 'value']*2
    table_head = [head1_row, head2_row]
    
    n_augsets = len(augmentation_set_numbers_list)
    
    table_dict = {}
    for i_augset, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        table = [[] for i in range(n_top_values)]
        for i_cval in range(n_sens_vals):
            cur_vals = loaded_mean_activations_list[i_cval][i_augset].flatten()
            ind = cur_vals.argsort()[::-1]
            cur_vals = cur_vals[ind]
            for i_row in range(n_top_values):
                id_class = ind[i_row]
                cval = f'{cur_vals[i_row]:.3f}'
                if cval.startswith('0'):
                    cval = cval.split('.')[-1]
                    cval = f'.{cval}'
                table[i_row] += [
                    id_class,
                    ind2class_dict[id_class],
                    cval,
                ]
        table = pd.DataFrame(
            data=table,
            columns=pd.MultiIndex.from_arrays(table_head)
        )
        table_dict[augmentation_set_number] = table
    return table_dict

def extract_per_class_metrics(
    prediction_results_path,
    no_aug_key='original',
    y_true_key='true_labels',
    label_name='labels',
    predictions_name='predictions',
):
    
    with h5py.File(prediction_results_path, 'r') as hf:
        #augmentation_names = list(hf.keys())
        #augmentation_names.remove(y_true_key)
        y_true = np.array(hf[y_true_key][label_name])
        Nsamples = y_true.size
        print(f'Nsamples={Nsamples}')
        y_true = y_true[:Nsamples]
        #y_true = y_true.reshape((Nsamples, 1))

        current_group = hf[no_aug_key][f'iaug={no_aug_key}']
        n_preds = current_group.attrs['shape'][-1]
        shape = (Nsamples, -1, n_preds)
        y_pred = np.array(current_group[predictions_name]).reshape(shape)
        #print(y_pred.shape)
    #y_pred = y_pred[:, 0, 0] # :, 1 sample - 1 sample, top-1
    
    uni_labels_pred, pred_counts = np.unique(y_pred.flatten(), return_counts=True)
    
    per_class_errors = []
    pred_counts_list = []
    uni_labels = np.unique(y_true)
    for i in range(len(uni_labels)):
        ind = np.where(y_true == uni_labels[i])[0]
        per_class_errors.append(
            (y_true[ind] != y_pred[ind, 0, 0]).mean()
        )
        ind2 = np.where(uni_labels_pred == uni_labels[i])[0]
        pred_counts_list.append(
            pred_counts[ind2]
        )
    pred_counts_list = np.array(pred_counts_list, dtype='f')
    pred_counts_list /= pred_counts_list.sum()
    per_class_errors = np.array(per_class_errors)
    return per_class_errors, pred_counts_list[:, 0], uni_labels




def jaccard_index_routine(
    sensitivity_topn_set,
    predicted_labels,
):
    Nsamples = len(predicted_labels)
    jacc_ind = []
    for i in range(Nsamples):
        tmp = []
        for j in range(predicted_labels.shape[1]):
            cur_jac_ind_val = len(
                sensitivity_topn_set.intersection(
                    set(predicted_labels[i, j])
                )
            )
            cur_jac_ind_val = cur_jac_ind_val / len(
                sensitivity_topn_set.union(
                    set(predicted_labels[i, j])
                )
            )
            tmp.append(cur_jac_ind_val)
        jacc_ind.append(np.mean(tmp))
    jacc_ind = np.mean(jacc_ind)
    return jacc_ind

def extract_jaccard(
    prediction_results_path,
    sensitivity_topn,
    augmentation_names,
    no_aug_key='original',
    y_true_key='true_labels',
    label_name='labels',
    predictions_name='predictions',
):
    n_augs = len(augmentation_names)
    
    results_jacc = {}
    results_jacc_aug_no_mask = {}
    results_jacc_orig_no_mask = {}

    with h5py.File(prediction_results_path, 'r') as hf:
        augmentation_names = list(hf.keys())
        # print(keys)
        augmentation_names.remove(y_true_key)
        y_true = np.array(hf[y_true_key][label_name])
        Nsamples = y_true.size
        print(f'Nsamples={Nsamples}')
        y_true = y_true[:Nsamples]
        y_true = y_true.reshape((Nsamples, 1))

        current_group = hf[no_aug_key][f'iaug={no_aug_key}']
        n_preds = current_group.attrs['shape'][-1]
        shape = (Nsamples, -1, n_preds)
        y_pred = np.array(current_group[predictions_name]).reshape(shape)
        for i_aug_in, augmentation_input_name in enumerate(sensitivity_topn):
            jacc_ind = jaccard_index_routine(
                sensitivity_topn[augmentation_input_name],
                y_pred,
            )
            results_jacc_orig_no_mask[augmentation_input_name] = jacc_ind
            
        for i_aug_in, augmentation_input_name in enumerate(augmentation_names):
            augmentation_mask_config_str = f'iaug={no_aug_key}'
            #for x in hf[augmentation_input_name][augmentation_mask_config_str]:
            #    print(x)
            for augmentation_sensitivity_name in sensitivity_topn:
                current_group = hf[augmentation_input_name][augmentation_mask_config_str]
                n_preds = current_group.attrs['shape'][-1]
                shape = (Nsamples, -1, n_preds)
                y_pred = np.array(current_group[predictions_name]).reshape(shape)
                jacc_ind = jaccard_index_routine(
                    sensitivity_topn[augmentation_sensitivity_name],
                    y_pred,
                )

                results_jacc_aug_no_mask[
                    f'{augmentation_input_name}::{augmentation_sensitivity_name}'
                ] = jacc_ind
            
            for augmentation_mask_config_str in hf[augmentation_input_name]:
                #print(augmentation_mask_name)
                augmentation_mask_name = augmentation_mask_config_str.split('iaug=')[1]
                #augmentation_mask_name = augmentation_mask_name.split('=')[1]
                augmentation_mask_name = augmentation_mask_name.split('_invert')[0]
                if augmentation_mask_name not in sensitivity_topn:
                    continue
                current_group = hf[augmentation_input_name][augmentation_mask_config_str]
                #shape = [Nsamples, -1] + list(hf[x][y].attrs['shape'])
                #shape = [Nsamples, -1, 5]
                n_preds = current_group.attrs['shape'][-1]
                shape = (Nsamples, -1, n_preds)
                y_pred = np.array(current_group[predictions_name]).reshape(shape)
                #shapes.add(tuple(shape))
                jacc_ind = jaccard_index_routine(
                    sensitivity_topn[augmentation_mask_name],
                    y_pred,
                )

                results_jacc[
                    f'{augmentation_input_name}::{augmentation_mask_config_str}'
                ] = jacc_ind

    return results_jacc, results_jacc_aug_no_mask, results_jacc_orig_no_mask
                
def collect_featured_measurements_jac(
    jaccard_results_dict,
    augmentation_set_numbers_list,
    alphas,
    percentiles,
    inverts,
    no_aug_key='original',
):
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=False,
    )
    
    n_features = len(alphas)*len(percentiles) + len(inverts)
    alpha_perc_feature_keys = []
    for alpha, perc in itertools.product(alphas, percentiles):
        alpha_perc_feature_keys.append(f'alpha={alpha:.2f}_perc={perc:.2f}')
    invert_feature_keys = []
    for invert in inverts:
        invert_feature_keys.append(f'invert={invert}')
    
    featured_measurements = {}
    featured_measurements_no_aug_input = {}
    #measurements_no_mask = {}
    
    for aug_set in augmentation_set_numbers_list:
        aug_names = augmentation_names_dict[aug_set]
        N_aug = len(aug_names)
        
        fm = np.zeros((N_aug, N_aug, n_features))
        fm_nai = np.zeros((N_aug, n_features))
        m_nm = np.zeros((N_aug, N_aug))
        
        #m_nm[-1] = accuracy_results_dict[f'{no_aug_key}::iaug={no_aug_key}']
        for i, aug1 in enumerate(aug_names):
            #m_nm[i] = accuracy_results_dict[f'{aug1}::iaug={no_aug_key}']
            for j, aug2 in enumerate(aug_names):
                for k, feature_key in enumerate(alpha_perc_feature_keys):
                    fm[i, j, k] = jaccard_results_dict[f'{aug1}::{feature_key}_iaug={aug2}']
                    if i == 0:
                        fm_nai[j, k] = jaccard_results_dict[f'{no_aug_key}::{feature_key}_iaug={aug2}']
                k_offset = len(alpha_perc_feature_keys)
                for k, feature_key in enumerate(invert_feature_keys):
                    fm[i, j, k_offset+k] = jaccard_results_dict[f'{aug1}::iaug={aug2}_{feature_key}']
                    if i == 0:
                        fm_nai[j, k_offset+k] = jaccard_results_dict[f'{no_aug_key}::iaug={aug2}_{feature_key}']
        #m_nm = pd.DataFrame(data=m_nm[None, :], columns=aug_names+[no_aug_key])
        featured_measurements[aug_set] = fm
        featured_measurements_no_aug_input[aug_set] = fm_nai
        #measurements_no_mask[aug_set] = m_nm
        del fm, fm_nai;#, m_nm;
    return featured_measurements, featured_measurements_no_aug_input#, measurements_no_mask
                
def get_sensitivity_topn_dict(
    loaded_sensitivity_values_dict_list,
    classifying_layer_name,
    augmentation_set_numbers_list,
    top_n=5,
    extract_auxilliary_names=False,
):
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    n_aug = len(augmentation_names)
    
    rv = []
    for loaded_sensitivity_values_dict in loaded_sensitivity_values_dict_list:
        sensitivity_topn = {}
        for i_aug, aug_name in enumerate(augmentation_names):
            ind = loaded_sensitivity_values_dict[
                classifying_layer_name
            ][i_aug].argsort()[::-1]
            sensitivity_topn[aug_name] = set(ind[:top_n])
        rv.append(sensitivity_topn)
    return rv                

def extract_table_no_masking(
    results_jacc_aug_no_mask,
    results_jacc_orig_no_mask,
    augmentation_set_numbers_list,
    extract_auxilliary_names=False,
    n_round_digits=4,
    no_aug_key='original',
):

    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    n_aug = len(augmentation_names)

    tmp = np.empty((n_aug+1, n_aug))
    for j, aug_name in enumerate(augmentation_names):
        tmp[0, j] = results_jacc_orig_no_mask[aug_name]
    for i in range(n_aug):
        for j in range(n_aug):
            key = f'{augmentation_names[i]}::{augmentation_names[j]}'
            tmp[i+1, j] = results_jacc_aug_no_mask[key]
    tmp = tmp.round(n_round_digits)
    tmp_df = pd.DataFrame(
        data=tmp,
        columns=pd.MultiIndex.from_arrays(
            [
                [
                    'SA variable used for extracting the most '
                    'sensitive classes from the classifying layer'
                ]*n_aug,
                augmentation_names
            ]
        ), 
        index=[no_aug_key]+augmentation_names
    )
    tmp_df.index.name = 'Augmentation of input'
    return tmp_df

def extract_table_self_masking(
    featured_measurements,
    augmentation_set_numbers_list,
    alphas,
    percentiles,
    inverts,
    n_round_digits=4,
    extract_auxilliary_names=False,
):
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    n_aug = len(augmentation_names)

    n_features = len(alphas)*len(percentiles) + len(inverts)
    tmp = np.empty((n_aug, n_features))
    i_aug = 0
    for aug_set in augmentation_set_numbers_list:
        for i in range(min(featured_measurements[aug_set].shape[:2])):
            tmp[i_aug, :] = featured_measurements[aug_set][i, i]
            i_aug += 1

    alpha_perc_feature_keys = []
    for alpha, perc in itertools.product(alphas, percentiles):
        alpha_perc_feature_keys.append(f'alpha={alpha:.2f}_perc={perc:.2f}')
    invert_feature_keys = []
    
    header = []
    header.append(
        [
            'Mask configuration (SA variable is the same as augmentation)'
        ]*n_features
    )
    tmp_h = []
    for a in alphas:
        tmp_h += [f'alpha={a}']*len(percentiles)
    tmp_h += [f'inv={int(cinv)}' for cinv in inverts]
    header.append(tmp_h)
    tmp_h = list(map(lambda x: f'q={x}', percentiles))*len(alphas)
    tmp_h += ['']*len(inverts)
    header.append(tmp_h)
    
    for invert in inverts:
        invert_feature_keys.append(f'invert={invert}')
    
    tmp_df = pd.DataFrame(
        data=tmp.round(n_round_digits),
        index=augmentation_names,
        columns=pd.MultiIndex.from_arrays(
            header
        ),
    )
    tmp_df.index.name = 'Augmentation of input'
    return tmp_df
    
    
    
    

import functools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import correlation.compute
import sensitivity_analysis.visualize
import preparation.visualize

_nan_policy_list = ('all', 'any')


def single_plot_corr_variables(
    ax,
    C,
    variable_names,
    cmap='RdBu_r',
    title=None,
):
    n_vals = len(variable_names)
    x_arr = range(n_vals)

    df_cmat = pd.DataFrame(
        C, columns=variable_names, index=variable_names
    )
    sns.heatmap(
        df_cmat,
        ax=ax,
        annot=True,
        cbar=False,
        fmt='.2f',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        #mask=mask,
    )
    ax.set_xticklabels(
        variable_names,
        #fontsize=10,
        rotation=25,
        #loc='center',
        #labelpad=5
    )
    '''
    ax.imshow(C, cmap=cmap, vmin=-1, vmax=1.)
    ax.set_yticks(x_arr)
    ax.set_yticklabels(
        variable_names,
        #fontsize=10,
        #rotation=10,
        #loc='center',
        #labelpad=5
    )
    ax.set_xticks(x_arr)
    ax.set_xticklabels(
        variable_names,
        rotation=90
    )
    '''
    ax.xaxis.set_label_position('top')
    ax.set_title(title)
    
def single_plot_corr_hist_variables(
    ax,
    vals,
    title=None,
    bins=50,
    xlabel='Values',
    print_left_ylabel=True,
    print_right_ylabel=True,
    color_cumul='salmon',
    color_bins='firebrick',
):
    ax2 = ax.twinx()
    ax2.hist(
        vals[~np.isnan(vals)],
        bins=bins,
        cumulative=True,
        alpha=0.5,
        color=color_cumul,
    )
    if print_right_ylabel:
        ax2.set_ylabel('Cumul.counts', rotation=-90, labelpad=10)
    else:
        ax2.set_yticks([])
        ax2.set_yticklabels([])
    
    ax.hist(
        vals[~np.isnan(vals)],
        bins=bins,
        cumulative=False,
        alpha=0.75,
        color=color_bins,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.5)
    if print_left_ylabel:
        ax.set_ylabel('Counts')
        
def single_plot_corr_modules(
    ax,
    i_ax,
    mean_vals,
    module_names,
    std_vals=None,
    title=None,
    cmap='twilight',
):
    n_modules, n_corrcoef = mean_vals.shape
    if std_vals is not None:
        n_std_modules = std_vals.shape[0]
    x_arr = range(n_modules)
    #colors = plt.cm.nipy_spectral_r(np.linspace(0, 1, n_corrcoef))
    #colors = plt.cm.gist_heat_r(np.linspace(0, 1, n_corrcoef))
    colors = getattr(plt.cm, cmap)(np.linspace(0, 1, n_corrcoef))
    for i in range(n_corrcoef):
        ax.plot(
            x_arr,
            mean_vals[:, i],
            '*-',
            color=colors[i],
            alpha=0.5
        )
        if std_vals is not None:
            ax.fill_between(
                x_arr[:n_std_modules],
                y1=mean_vals[:n_std_modules, i]+std_vals[:, i],
                y2=mean_vals[:n_std_modules, i]-std_vals[:, i],
                color=colors[i],
                alpha=0.05,
            )
    ax.set_xticks(x_arr)
    ax.set_xticklabels(
        module_names,
        rotation=25
    )
    ax.xaxis.set_label_position('bottom')
    ax.set_title(title)
    ax.grid(alpha=0.5)
    if i_ax == 0:
        ax.set_ylabel('Correlation coef.')
    ax.set_ylim((-1.1, 1.1))
    

def get_paired_correlations_fixed_values(
    values_fnms_dict,
    activations_dirname,
    network_modules_list,
    values_names_list,
    augmentation_set_numbers_list,
    shpv_group_indices_dict,
    n_conv_modules,
    corr_type='spearman',
    extract_auxilliary_names=False,
    nan_mask_policy='all',
):
    assert nan_mask_policy in _nan_policy_list

    n_vals = len(values_names_list)
    n_modules = len(network_modules_list)

    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    n_aug = len(augmentation_names)
    ind = np.triu_indices(n=n_aug, k=1)
    
    cormat_means = np.empty((n_modules, n_vals, n_aug, n_aug))
    cormat_std = np.empty((n_conv_modules, n_vals, n_aug*(n_aug-1)//2))
    cormat_nan_masks = np.empty((n_modules, n_vals, n_aug, n_aug))
    for i_mn, module_name in enumerate(network_modules_list):
        for i_row, row_name in enumerate(values_names_list):
            value_key = preparation.visualize._values_keys_dict[row_name]
            dataset_part = None
            if 'rbscc' in row_name:
                _, dataset_part = row_name.split(' ')
                dataset_part = dataset_part.replace('(', '').replace(')', '')
            c_values_name = preparation.visualize._values_names_dict[value_key]
            values = []
            for i_col in range(n_aug):
                aug_name = augmentation_names[i_col]
                slice_num = i_col
                for augmentation_set_number in augmentation_set_numbers_list:
                    L = len(augmentation_names_dict[augmentation_set_number])
                    if slice_num >= L:
                        slice_num -= L
                    else:
                        break
                current_values = preparation.visualize.get_conv2d_unit_values(
                    values_fnms_dict,
                    activations_dirname,
                    module_name,
                    values_key=value_key,
                    values_name=c_values_name,
                    augmentation_set_number=augmentation_set_number,
                    dataset_part=dataset_part,
                    slice_num=slice_num,
                    values_func=None if row_name not in preparation.visualize._values_funcs else lambda x: (
                        preparation.visualize._values_funcs[row_name](
                            x,
                            shpv_group_indices_dict[augmentation_set_number]
                        )
                    ),
                    shpv_normalize=True,
                )
                values.append(current_values)
            values = np.array(values)
            #print(values.min(), values.max())
            n_units = values.shape[1]
            values = values.reshape((n_aug, n_units, -1))
            C_mean, C_std, C_nan_mask = correlation.compute.compute_unitwise_correlation(
                values, None, corr_type
            )
            del values;
            cormat_means[i_mn, i_row, :, :] = C_mean
            if nan_mask_policy == 'all':
                cormat_nan_masks[i_mn, i_row, :, :] = C_nan_mask.all(axis=0)
            elif nan_mask_policy == 'any':
                cormat_nan_masks[i_mn, i_row, :, :] = C_nan_mask.any(axis=0)
            if i_mn < n_conv_modules:
                cormat_std[i_mn, i_row, :] = C_std[ind]
            
    return cormat_means, cormat_std, augmentation_names, cormat_nan_masks

def get_paired_cross_correlations_fixed_values(
    values_fnms_dict,
    activations_dirname,
    network_modules_list,
    values_names_list,
    augmentation_set_numbers_list,
    shpv_group_indices_dict,
    n_conv_modules,
    corr_type='spearman',
    extract_auxilliary_names=True,
    nan_mask_policy='all',
):
    assert nan_mask_policy in _nan_policy_list
    
    n_modules = len(network_modules_list)
    n_vals = len(values_names_list)
    n_vals_pairs = n_vals*(n_vals-1)//2

    augmentation_and_auxilliary_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_and_auxilliary_names = functools.reduce(
        lambda x, y: x+y, augmentation_and_auxilliary_names_dict.values()
    )

    n_aug_groups = len(augmentation_set_numbers_list)
    n_aug_aux = len(augmentation_and_auxilliary_names)

    ind = np.triu_indices(n=n_aug_aux, k=1)

    cormat_means = np.empty((n_modules, n_vals_pairs, n_aug_aux, n_aug_aux))
    cormat_nanmask = np.empty((n_modules, n_vals_pairs, n_aug_aux, n_aug_aux))
    cormat_std = np.empty((n_conv_modules, n_vals_pairs, n_aug_aux*(n_aug_aux-1)//2))
    cross_values_names_list = []

    dataset_part = None
    for i_mn, module_name in enumerate(network_modules_list):
        for i_row, row_name in enumerate(values_names_list[:-1]):
            value_key = preparation.visualize._values_keys_dict[row_name]
            c_values_name = preparation.visualize._values_names_dict[value_key]
            for i_row2, row_name2 in enumerate(values_names_list[i_row+1:]):
                value_key2 = preparation.visualize._values_keys_dict[row_name2]
                c_values_name2 = preparation.visualize._values_names_dict[value_key2]

                values, values2 = [], []
                for i_col in range(n_aug_aux):
                    aug_aux_name = augmentation_and_auxilliary_names[i_col]
                    slice_num = i_col
                    for augmentation_set_number in augmentation_set_numbers_list:
                        L = len(augmentation_and_auxilliary_names_dict[augmentation_set_number])
                        if slice_num >= L:
                            slice_num -= L
                        else:
                            break
                    current_values = preparation.visualize.get_conv2d_unit_values(
                        values_fnms_dict,
                        activations_dirname,
                        module_name,
                        values_key=value_key,
                        values_name=c_values_name,
                        augmentation_set_number=augmentation_set_number,
                        dataset_part=dataset_part,
                        slice_num=slice_num,
                        values_func=None if row_name not in preparation.visualize._values_funcs else lambda x: (
                            preparation.visualize._values_funcs[row_name](
                                x,
                                shpv_group_indices_dict[augmentation_set_number]
                            )
                        ),
                        shpv_normalize=True,
                    )
                    values.append(current_values)
                    current_values2 = preparation.visualize.get_conv2d_unit_values(
                        values_fnms_dict,
                        activations_dirname,
                        module_name,
                        values_key=value_key2,
                        values_name=c_values_name2,
                        augmentation_set_number=augmentation_set_number,
                        dataset_part=dataset_part,
                        slice_num=slice_num,
                        values_func=None if row_name2 not in preparation.visualize._values_funcs else lambda x: (
                            preparation.visualize._values_funcs[row_name2](
                                x,
                                shpv_group_indices_dict[augmentation_set_number]
                            )
                        ),
                        shpv_normalize=True,
                    )
                    values2.append(current_values2)

                values = np.array(values)
                values2 = np.array(values2)
                #print(values.min(), values2.min(), values.max(), values2.max())
                n_units = values.shape[1]
                values = values.reshape((n_aug_aux, n_units, -1))
                values2 = values2.reshape((n_aug_aux, n_units, -1))
                C_mean, C_std, C_nan_mask = correlation.compute.compute_unitwise_correlation(
                    values, values2, corr_type
                )
                del values;
                i_row_ind = i_row*(n_vals-1)+i_row2
                cormat_means[i_mn, i_row_ind, :, :] = C_mean
                if i_mn < n_conv_modules:
                    cormat_std[i_mn, i_row_ind, :] = C_std[ind]
                if nan_mask_policy == 'all':
                    cormat_nanmask[i_mn, i_row_ind, :, :] = C_nan_mask.all(axis=0)
                elif nan_mask_policy == 'any':
                    cormat_nanmask[i_mn, i_row_ind, :, :] = C_nan_mask.any(axis=0)
                if i_mn == 0:
                    cross_values_names_list.append(f'{row_name}--{row_name2}')
    
    return cormat_means, cormat_std, augmentation_and_auxilliary_names, cross_values_names_list, cormat_nanmask

def get_cross_correlations_single_value_fixed_values(
    values_fnms_dict,
    activations_dirname,
    network_modules_list,
    values_names_list,
    augmentation_set_numbers_list,
    shpv_group_indices_dict,
    n_conv_modules,
    values_corr_pairs_dict,
    corr_values_names_dict,
    cov_values_funcs,
    corr_type='spearman',
    extract_auxilliary_names=True,
):

    n_vals = len(values_names_list)
    n_modules = len(network_modules_list)

    augmentation_and_auxilliary_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=True,
    )
    augmentation_and_auxilliary_names = functools.reduce(
        lambda x, y: x+y, augmentation_and_auxilliary_names_dict.values()
    )
    n_aug_groups = len(augmentation_set_numbers_list)
    n_aug_aux = len(augmentation_and_auxilliary_names)

    corr_vals_dict, corr_vals_nanmask_dict = {}, {}
    mean_stats = np.empty((n_modules, n_vals, n_aug_aux))
    std_stats = np.empty((n_conv_modules, n_vals, n_aug_aux))

    for i_mn, module_name in enumerate(network_modules_list):
        corr_values, corr_values_nanmask = [], []
        for i_row, row_name in enumerate(values_names_list):
            value_key = preparation.visualize._values_keys_dict[row_name]
            dataset_part = None
            c_values_name = preparation.visualize._values_names_dict[value_key]
            cval_corr_values, cval_corr_values_nanmask = [], []
            for i_col in range(n_aug_aux):
                aug_aux_name = augmentation_and_auxilliary_names[i_col]
                slice_num = i_col
                for augmentation_set_number in augmentation_set_numbers_list:
                    L = len(augmentation_and_auxilliary_names_dict[augmentation_set_number])
                    if slice_num >= L:
                        slice_num -= L
                    else:
                        break
                current_values = preparation.visualize.get_conv2d_unit_values(
                    values_fnms_dict,
                    activations_dirname,
                    module_name,
                    values_key=value_key,
                    values_name=c_values_name,
                    augmentation_set_number=augmentation_set_number,
                    dataset_part=dataset_part,
                    slice_num=slice_num,
                    values_func=None if row_name not in preparation.visualize._values_funcs else lambda x: (
                        preparation.visualize._values_funcs[row_name](
                            x,
                            shpv_group_indices_dict[augmentation_set_number]
                        )
                    ),
                    shpv_normalize=True,
                )
                ##
                cov_value_key = values_corr_pairs_dict[value_key]
                cov_values_name = corr_values_names_dict[cov_value_key]
                cov_values = preparation.visualize.get_conv2d_unit_values(
                    values_fnms_dict,
                    activations_dirname,
                    module_name,
                    cov_value_key,
                    values_name=cov_values_name,
                    augmentation_set_number=augmentation_set_number,
                    dataset_part=dataset_part,
                    slice_num=0,
                    values_func=None if cov_values_funcs is None else cov_values_funcs[cov_value_key][cov_values_name],
                    shpv_normalize=False,
                )
                #print(current_values.min(), current_values.max(), cov_values.min(), cov_values.max())
                current_corvals, current_nan_mask = correlation.compute.compute_correlation_cov_aug(
                    cov_values, current_values, corr_type
                )
                cval_corr_values.append(current_corvals)
                cval_corr_values_nanmask.append(current_nan_mask)
                if i_mn < n_conv_modules:
                    mean_stats[i_mn, i_row, i_col] = current_corvals.mean()
                    std_stats[i_mn, i_row, i_col] = current_corvals.std(ddof=1)
                else:
                    mean_stats[i_mn, i_row, i_col] = current_corvals                
                del current_values, cov_values;
            corr_values.append(cval_corr_values)
            corr_values_nanmask.append(cval_corr_values_nanmask)
        corr_vals_dict[module_name] = np.array(corr_values)
        corr_vals_nanmask_dict[module_name] = np.array(corr_values_nanmask)
    return corr_vals_dict, mean_stats, std_stats, augmentation_and_auxilliary_names, corr_vals_nanmask_dict
                
        
        

def plot_paired_correlations_fixed_values(
    cormat_means,
    augmentation_names,
    network_modules_list,
    values_names_list,
    figsize=None,
    plot_colorbar=True,
    show=True,
    cmap='RdBu_r',
    save_dirname=None,
    save_filename_base=None,
):

    n_vals = len(values_names_list)
    shorten_augmentation_names = list(
        map(
            lambda x: dict(
                **preparation.visualize._shorten_augnames_dict,
                **preparation.visualize._shorten_variables_dict
            )[x],
            augmentation_names
        )
    )
    n_aug = len(augmentation_names)

    if plot_colorbar:
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.25*10, h=0.005*10,
            vmin=-1, vmax=1, dv=0.2, ticks=None,
            label='', cmap=cmap
        )
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_colorbar.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()

        #dataset_part = None
        #fig, lgs, current_gs = preparation.visualize.init_geometry(
        #    fig, gs, n_vals, n_aug_aux, figsize, gs_size
        #)

    for i_mn, module_name in enumerate(network_modules_list):
        fig, ax = plt.subplots(1, n_vals, figsize=figsize, sharey=True)
        for i_row, row_name in enumerate(values_names_list):
            single_plot_corr_variables(
                ax[i_row],
                cormat_means[i_mn, i_row, :, :],
                shorten_augmentation_names,
                cmap=cmap,
                title=row_name
            )
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            print(f'\n\t\t\t{module_name}\n')
            plt.tight_layout()
            plt.show()
        else:
            plt.clf()

def plot_hist_std_paired_correlations_fixed_values(
    cormat_std,
    augmentation_names,
    network_modules_list,
    values_names_list,
    n_conv_modules,
    figsize=None,
    show=True,
    color_cumul='salmon',
    color_bins='firebrick',
):
    
    n_vals = len(values_names_list)
    n_aug = len(augmentation_names)

    for i_mn, module_name in enumerate(network_modules_list[:n_conv_modules]):
        print(f'\n\t\t\t{module_name}\n')
        fig, ax = plt.subplots(1, n_vals, figsize=figsize, sharey=True)
        for i_row, row_name in enumerate(values_names_list):
            single_plot_corr_hist_variables(
                ax[i_row],
                cormat_std[i_mn, i_row, :],
                title=row_name,
                xlabel='Standard deviation values',
                print_left_ylabel=(i_row == 0),
                print_right_ylabel=(i_row == n_vals-1),
                color_cumul=color_cumul,
                color_bins=color_bins,
            )
        if show:
            plt.show()

            
def plot_consolidated_paired_correlations_fixed_values(
    cormat_means,
    cormat_std,
    network_modules_list,
    values_names_list,
    figsize=None,
    show=True,
    cmap='twilight',
):
    n_vals = len(values_names_list)
    n_modules = len(network_modules_list)
    _, _, n_aug, _ = cormat_means.shape
    ind = np.triu_indices(n=n_aug, k=1)
    #tmp_means = np.empty((n_modules, n_vals, n_aug*(n_aug-1)//2))
    #tmp_means[:, :, :] = cormat_means[:, :, ind[0], ind[1]]

    fig, ax = plt.subplots(1, n_vals, figsize=figsize, sharey=True)
    for i_row, row_name in enumerate(values_names_list):
        single_plot_corr_modules(
            ax[i_row],
            i_row,
            cormat_means[:, i_row, ind[0], ind[1]],
            #tmp_means[:, i_row, :],
            network_modules_list,
            std_vals=cormat_std[:, i_row, :],
            title=row_name,
            cmap=cmap,
        )
    if show:
        plt.show()
        
        
def plot_cross_correlations_single_value_fixed_values(
    corr_vals_dict,
    augmentation_and_auxilliary_names,
    network_modules_list,
    values_names_list,
    n_conv_modules,
    figsize=None,
    plot_colorbar=True,
    show=True,
    cmap='RdBu_r',
    save_dirname=None,
    save_filename_base=None,
):
    n_vals = len(values_names_list)
    n_modules = len(network_modules_list)
    n_aug_aux = len(augmentation_and_auxilliary_names)
    
    if show and plot_colorbar:
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.25*10, h=0.005*10,
            vmin=-1, vmax=1, dv=0.2, ticks=None,
            label='', cmap=cmap
        )
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_colorbar.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()

    for i_mn, module_name in enumerate(network_modules_list):
        fig, ax = plt.subplots(n_vals, n_aug_aux, figsize=figsize, sharey=True)
        for i_row, row_name in enumerate(values_names_list):
            value_key = preparation.visualize._values_keys_dict[row_name]
            c_values_name = preparation.visualize._values_names_dict[value_key]
            cval_corr_values = []
            for i_col in range(n_aug_aux):
                aug_aux_name = augmentation_and_auxilliary_names[i_col]
                if i_mn < n_conv_modules:
                    ax[i_row, i_col].imshow(
                        corr_vals_dict[module_name][i_row, i_col], cmap=cmap, vmin=-1, vmax=1
                    )
                else:
                    ax[i_row, i_col].imshow(
                        corr_vals_dict[module_name][i_row, i_col].reshape((1, 1)), cmap=cmap, vmin=-1, vmax=1
                    )
                ax[i_row, i_col] = preparation.visualize.ax_disable_ticks_ticklabels(ax[i_row, i_col])
                if i_col == 0:
                    ax[i_row, i_col].set_ylabel(row_name)
                if i_row == 0:
                    ax[i_row, i_col].set_xlabel(aug_aux_name)
                    ax[i_row, i_col].xaxis.set_label_position('top')
                #ax[i_row, i_col].set_title(module_name)
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            #plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            print(f'\n\t\t\t{module_name}\n')
            #plt.tight_layout()
            plt.show()
        else:
            plt.clf()
                
def plot_consolidated_cross_correlations_single_value_fixed_values(
    cormat_means,
    cormat_std,
    network_modules_list,
    values_names_list,
    figsize=None,
    show=True,
    cmap='twilight_r',
):
    n_vals = len(values_names_list)
    n_modules = len(network_modules_list)
    _, _, n_aug = cormat_means.shape
    #tmp_means = np.empty((n_modules, n_vals, n_aug*(n_aug-1)//2))
    #tmp_means[:, :, :] = cormat_means[:, :, ind[0], ind[1]]

    fig, ax = plt.subplots(1, n_vals, figsize=figsize, sharey=True)
    for i_row, row_name in enumerate(values_names_list):
        single_plot_corr_modules(
            ax[i_row],
            i_row,
            cormat_means[:, i_row, :],
            network_modules_list,
            std_vals=cormat_std[:, i_row, :],
            title=row_name,
            cmap=cmap
        )
    if show:
        plt.show()
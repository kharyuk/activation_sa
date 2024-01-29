import itertools
import warnings
import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.patches
import matplotlib.pyplot as plt

import prediction.utils
from . import visualize


def unit_mean_pixelwise_correlation_patterns(
    mean_cormat_hsv,
    mean_cormat_orig,
    network_modules,
    augmentation_set_number,
    chans_name_list,
    value_name_list,
    extract_auxilliary_names=True,
    cmap='RdBu_r',
    figsize=(12, 12),
    show=True,
    no_chan_name='original',
    save_dirname=None,
    save_filename_base=None,
):
    
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
    n_layers = len(mean_cormat_hsv)
    assert n_layers == len(mean_cormat_orig)
    n_input_chans = len(chans_name_list)
    n_vals = len(value_name_list)
    for i_layer in range(n_layers):
        fig, ax = plt.subplots(
            n_input_chans+1, n_vals, figsize=figsize, sharey=True, sharex=True
        )
        for i_val, val_name in enumerate(value_name_list):
            for i_ch, chan_name in enumerate(chans_name_list):
                tmp = mean_cormat_hsv[i_layer, i_ch, i_val, :, :]
                cax = ax[i_ch, i_val]
                df_cmat = pd.DataFrame(
                    tmp[1:, :-1], columns=c_augaux_names[:-1], index=c_augaux_names[1:]
                )
                sns.heatmap(
                    df_cmat,
                    ax=cax,
                    annot=True,
                    cbar=False,
                    fmt='.2f',
                    cmap=cmap,
                    vmin=-1,
                    vmax=1,
                    mask=mask,
                )
                cax.tick_params(
                    axis='y',
                    which='both',
                    left=False,
                    #labelleft=False,
                )
                if i_val == 0:
                    cax.set_ylabel(f'{chan_name}')
                if i_ch == 0:
                    tmp = mean_cormat_orig[i_layer, i_val, :, :]
                    df_cmat = pd.DataFrame(
                        tmp[1:, :-1], columns=c_augaux_names[:-1], index=c_augaux_names[1:]
                    )
                    sns.heatmap(
                        df_cmat,
                        ax=ax[-1, i_val],
                        annot=True,
                        cbar=False,
                        fmt='.2f',
                        cmap=cmap,
                        vmin=-1,
                        vmax=1,
                        mask=mask,
                    )
                    xtick_labels = ax[-1, i_val].get_xticklabels()
                    if xtick_labels is not None:
                        ax[-1, i_val].set_xticklabels(xtick_labels, rotation=17.5)
                    if i_val == 0:
                        ax[-1, i_val].set_ylabel(f'{no_chan_name}')
                    
                    ax[-1, i_val].tick_params(
                        axis='y',
                        which='both',
                        left=False,
                    )

                    if i_ch == 0:
                        cax.set_title(f'({val_name})')
                #print(std_cor_mat.min(), std_cor_mat.max(), std_cor_mat2.min(), std_cor_mat2.max())
                
        plt.subplots_adjust(wspace=0, hspace=0)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{network_modules[i_layer]}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            print(f'\n\t\t\t{network_modules[i_layer]}')
            plt.tight_layout()
            plt.show()
        else:
            plt.clf()

            
def plot_hist_std_correlations_fixed_values(
    cormat_std_hsv,
    cormat_std_orig,
    augmentation_set_number,
    network_modules_list,
    chans_names_list,
    values_names_list,
    n_conv_modules,
    figsize=None,
    show=True,
    color_cumul='salmon',
    color_bins='firebrick',
    no_chan_name='original',
    extract_auxilliary_names=True,
):
    _, augmentation_names =  (
        prediction.utils.get_shortened_variable_names_single_augset(
            augmentation_set_number,
            extract_auxilliary_names=extract_auxilliary_names,
        )
    )
    n_chans = len(chans_names_list)
    n_vals = len(values_names_list)
    n_aug = len(augmentation_names)
    ind = np.triu_indices(n_aug, k=1)
    
    orig_flag = cormat_std_orig is not None

    for i_mn, module_name in enumerate(network_modules_list[:n_conv_modules]):
        print(f'\n\t\t\t{module_name}\n')
        fig, ax = plt.subplots(n_chans+int(orig_flag), n_vals, figsize=figsize, sharey=True, sharex=False)
        for i_row, row_name in enumerate(chans_names_list):
            for i_col, col_name in enumerate(values_names_list):
                visualize.single_plot_corr_hist_variables(
                    ax[i_row, i_col],
                    cormat_std_hsv[i_mn, i_row, i_col, ind[0], ind[1]],
                    title='' if i_row > 0 else f'({col_name})',
                    xlabel='',
                    print_left_ylabel=(i_col == 0),
                    print_right_ylabel=(i_col == n_vals-1),
                    color_cumul=color_cumul,
                    color_bins=color_bins,
                )
                if i_col == 0:
                    ax[i_row, i_col].set_ylabel(row_name)
                if (i_row == 0) and (orig_flag):
                    #print(cormat_std_orig[i_mn, i_col, ind[0], ind[1]].shape, n_aug)
                    visualize.single_plot_corr_hist_variables(
                        ax[-1, i_col],
                        cormat_std_orig[i_mn, i_col, ind[0], ind[1]],
                        title='',
                        xlabel='Standard deviation values',
                        print_left_ylabel=(i_col == 0),
                        print_right_ylabel=(i_col == n_vals-1),
                        color_cumul=color_cumul,
                        color_bins=color_bins,
                    )
                    if i_col == 0:
                        ax[-1, i_col].set_ylabel(no_chan_name)
        if show:
            plt.tight_layout()
            plt.show()

def plot_consolidated_paired_correlations_fixed_values(
    cormat_mean_hsv,
    cormat_std_hsv,
    cormat_mean_orig,
    cormat_std_orig,
    network_modules_list,
    chans_names_list,
    values_names_list,
    figsize=None,
    show=True,
    cmap='twilight',
    no_chan_name='original',
    vmin=-1,
):
    n_chans = len(chans_names_list)
    n_vals = len(values_names_list)
    n_modules = len(network_modules_list)
    _, _, _, n_aug, _ = cormat_std_hsv.shape
    ind = np.triu_indices(n=n_aug, k=1)
    #tmp_means = np.empty((n_modules, n_vals, n_aug*(n_aug-1)//2))
    #tmp_means[:, :, :] = cormat_means[:, :, ind[0], ind[1]]

    orig_flag = (cormat_mean_orig is not None) and (cormat_std_orig is not None)
    
    fig, ax = plt.subplots(n_chans+int(orig_flag), n_vals, figsize=figsize, sharey=True, sharex=True)
    for i_row, row_name in enumerate(chans_names_list):
        for i_col, col_name in enumerate(values_names_list):
            visualize.single_plot_corr_modules(
                ax[i_row, i_col],
                i_col,
                cormat_mean_hsv[:, i_row, i_col, ind[0], ind[1]],
                network_modules_list,
                std_vals=cormat_std_hsv[:, i_row, i_col, ind[0], ind[1]],
                title='' if i_row > 0 else f'({col_name})',
                cmap=cmap,
            )
            if i_col == 0:
                ax[i_row, i_col].set_ylabel(row_name)
            vmax = (
                cormat_mean_hsv[:, i_row, i_col, ind[0], ind[1]]
                + cormat_std_hsv[:, i_row, i_col, ind[0], ind[1]]
            ).max()
            ax[i_row, i_col].set_ylim((vmin, max(vmax, 1)))
            if (i_row == 0) and (orig_flag):
                visualize.single_plot_corr_modules(
                    ax[-1, i_col],
                    i_col,
                    cormat_mean_orig[:, i_col, ind[0], ind[1]],
                    network_modules_list,
                    std_vals=cormat_std_orig[:, i_col, ind[0], ind[1]],
                    title='',
                    cmap=cmap,
                )
                if i_col == 0:
                    ax[-1, i_col].set_ylabel(no_chan_name)
    if show:
        plt.tight_layout()
        plt.show()
        
        
def unit_mean_pixelwise_diff_correlation_patterns(
    mean_cormat_diff,
    network_modules,
    augmentation_set_number,
    chans_name_list,
    value_name_list,
    extract_auxilliary_names=True,
    cmap='RdBu_r',
    figsize=(12, 12),
    show=True,
):

    _, c_augaux_names =  (
        prediction.utils.get_shortened_variable_names_single_augset(
            augmentation_set_number,
            extract_auxilliary_names=extract_auxilliary_names,
        )
    )
    n_vars = len(c_augaux_names)
    c_n_chan_input = len(chans_name_list)
    mask = np.ones((n_vars-1, n_vars-1))
    mask_ind = np.tril_indices(n_vars-1, 0, m=n_vars-1)
    mask[mask_ind] = 0
    for i_layer in range(len(network_modules)):
        fig, ax = plt.subplots(
            len(chans_name_list), len(value_name_list), figsize=figsize, sharey=True, sharex=True
        )
        for i_val, row_val_name in enumerate(value_name_list):
            for i_ch, chan_name in enumerate(chans_name_list):
                cax = ax[i_ch, i_val]
                cor_mat = mean_cormat_diff[i_layer, i_ch, i_val, 1:, :-1]
                df_cmat = pd.DataFrame(
                    #mean_cormat_diff[i_layer, i_ch, i_val, 1:, :-1],
                    cor_mat,
                    columns=c_augaux_names[:-1],
                    index=c_augaux_names[1:],
                )
                cur_abs_err = cor_mat[mask_ind].mean()
                sns.heatmap(
                    df_cmat,
                    ax=cax,
                    annot=True,
                    cbar=False,
                    fmt='.2f',
                    cmap=cmap,
                    vmin=-1,
                    vmax=1,
                    mask=mask,
                )
                if i_val == 0:
                    cax.set_ylabel(f'{chan_name}')
                if i_ch == c_n_chan_input-1:
                    xtick_labels = ax[i_ch, i_val].get_xticklabels()
                    if xtick_labels is not None:
                        ax[i_ch, i_val].set_xticklabels(xtick_labels, rotation=17.5)
                title = f'mean abs.err.={cur_abs_err:.2f}'
                if i_ch == 0:
                    title = f'({row_val_name})\n{title}'
                cax.set_title(title)
                #print(std_cor_mat.min(), std_cor_mat.max(), std_cor_mat2.min(), std_cor_mat2.max(), cor_mat2.min(), cor_mat2.max())
        print(f'\n\t\t\t{network_modules[i_layer]}')
        if show:
            plt.tight_layout()
            plt.show()
        

def hsv_correlation_diff_patterns_among_units(
    mean_cormat,
    network_modules,
    chans_name_list,
    value_name_list,
    extract_auxilliary_names=True,
    use_diag=True,
    triangular=True,
    cmap='bone_r',
    figsize=(12, 12),
    show=True,
    save_path=None,
    cmap_bounds=None,
    cormat_nanmask=None,
):
    if cmap_bounds is None:
        vmin, vmax = 0, None
    else:
        vmin, vmax = cmap_bounds
    
    n_layers = len(network_modules)
    n_input_chans = len(chans_name_list)
    n_vals = len(value_name_list)
    assert mean_cormat.shape == (n_input_chans, n_vals, n_layers, n_layers)
    
    if use_diag:
        mask_shape = (n_layers, n_layers)
        mask = np.zeros(mask_shape)
        if triangular:
            mask_ind = np.triu_indices(n_layers, k=1)
            mask = np.zeros(mask_shape)
            mask[mask_ind] = 1
    else:
        mask_shape = (n_layers-1, n_layers-1)
        if triangular:
            mask = np.zeros(mask_shape)
            mask_ind = np.triu_indices(n_layers-1, k=1)
            mask[mask_ind] = 1
        else:
            raise NotImplementedError

    fig, ax = plt.subplots(
        n_input_chans, n_vals, figsize=figsize, sharey=True, sharex=True
    )
    for i_val, row_val_name in enumerate(value_name_list):
        for i_ch, chan_name in enumerate(chans_name_list):
            cax = ax[i_ch, i_val]
            if use_diag:
                df_cmat = pd.DataFrame(
                    mean_cormat[i_ch, i_val, :, :],
                    columns=network_modules[:n_layers],
                    index=network_modules[:n_layers]
                )
                if cormat_nanmask is not None:
                    cur_cormat_nanmask = cormat_nanmask[i_ch, i_val, :, :]
            else:
                df_cmat = pd.DataFrame(
                    mean_cormat[i_ch, i_val, 1:, :-1],
                    columns=network_modules[:n_layers-1],
                    index=network_modules[1:n_layers]
                )
                if cormat_nanmask is not None:
                    cur_cormat_nanmask = cormat_nanmask[i_ch, i_val, 1:, :-1]
                    
            
            if cormat_nanmask is not None:
                cmap_inds = np.nonzero(cur_cormat_nanmask)
                n_nnz = len(cmap_inds[0])
                w, h = 1, 1
                for i_cmap_nnz in range(n_nnz):
                    x, y = cmap_inds[1][i_cmap_nnz], cmap_inds[0][i_cmap_nnz]
                    cax.add_patch(
                        matplotlib.patches.Rectangle(
                            (x, y), w, h, linewidth=0, fill=None, hatch='///'
                        )
                    )
            
            #cur_cmap = mpl.cm.get_cmap(cmap)
            #cur_cmap.set_bad('w', alpha=0.)
            sns.heatmap(
                df_cmat,
                ax=cax,
                annot=True,
                cbar=False,
                fmt='.2f',
                #cmap='pink_r',
                #cmap='gray_r',
                #cmap='gist_yarg',
                ##cmap='gist_heat_r',
                #cmap='gist_gray_r',
                #cmap='copper_r',
                cmap=cmap,
                #cmap='binary',
                ##cmap='afmhot_r',
                #cmap='YlOrRd',
                #cmap='Greys',
                #cmap='OrRd',
                #cmap='Oranges',
                ###cmap='Purples',
                ##cmap='RdGy',
                vmin=vmin,
                vmax=vmax,
                mask=mask,
            )
            #if cormat_nanmask is not None:
            #    cmap_inds = np.nonzero(cur_cormat_nanmask.flatten(order='C'))[0]
            #    print(cmap_inds)
            #    cur_cmap._init()
            #    cur_cmap._lut[cmap_inds, :] = np.array([149,238,58,255])/255.
            
            if i_val == 0:
                cax.set_ylabel(f'{chan_name}')
                ytick_labels = ax[i_ch, i_val].get_yticklabels()
                if ytick_labels is not None:
                    ax[i_ch, i_val].set_yticklabels(ytick_labels, rotation=0)
                
            if i_ch == n_input_chans-1:
                xtick_labels = ax[i_ch, i_val].get_xticklabels()
                if xtick_labels is not None:
                    ax[i_ch, i_val].set_xticklabels(xtick_labels, rotation=17.5)
            if i_ch == 0:
                title = f'({row_val_name})'
                cax.set_title(title)
    plt.subplots_adjust(wspace=0.05)#, hspace=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.clf()


def plot_hist_std_correlations_fixed_values_cross_layers(
    std_cormat,
    network_modules,
    chans_names_list,
    values_names_list,
    figsize=None,
    show=True,
    color_cumul='salmon',
    color_bins='firebrick',
    extract_auxilliary_names=True,
    use_diag=True,
    triangular=True,
):

    n_chans = len(chans_names_list)
    n_vals = len(values_names_list)
    n_layers = len(network_modules)
    assert std_cormat.shape == (n_chans, n_vals, n_layers, n_layers)
    
    if use_diag:
        if triangular:
            ind = np.triu_indices(n_layers, k=1)
        else:
            ind = list(
                map(
                    lambda x: x.flatten(),
                    np.indices((n_layers, n_layers), sparse=False)
                )
            )
    else:
        if triangular:
            ind = np.triu_indices(n_layers-1, k=1)
        else:
            raise NotImplementedError

    fig, ax = plt.subplots(n_chans, n_vals, figsize=figsize, sharey=True, sharex=False)
    for i_row, row_name in enumerate(chans_names_list):
        for i_col, col_name in enumerate(values_names_list):
            visualize.single_plot_corr_hist_variables(
                ax[i_row, i_col],
                std_cormat[i_row, i_col, ind[0], ind[1]],
                title='' if i_row > 0 else f'({col_name})',
                xlabel='',
                print_left_ylabel=(i_col == 0),
                print_right_ylabel=(i_col == n_vals-1),
                color_cumul=color_cumul,
                color_bins=color_bins,
            )
            if i_col == 0:
                ax[i_row, i_col].set_ylabel(row_name)
    if show:
        plt.tight_layout()
        plt.show()


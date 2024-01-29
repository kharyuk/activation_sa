import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns


from . import utils

def plot_spearman_correlation_masked_activations_original_input_vs_augmented(
    featured_measurements_dict,
    featured_measurements_no_aug_input_dict,
    augmentation_set_number,
    extract_auxilliary_names=False,
    figsize=(15, 5),
    show=True,
    cmap='Reds',
    save_path=None,
):
    augmentation_names, augmentation_and_auxilliary_names =  (
        utils.get_shortened_variable_names_single_augset(
            augmentation_set_number,
            extract_auxilliary_names=extract_auxilliary_names,
        )
    )
    
    n_augs = len(augmentation_names)
    n_augs_aux = len(augmentation_and_auxilliary_names)
    for i, M in enumerate(featured_measurements_dict.values()):
        if i == 0:
            n_features = M.shape[-1]
        assert M.shape == (n_augs, n_augs_aux, n_features)
    for M in featured_measurements_no_aug_input_dict.values():
        assert M.shape == (n_augs_aux, n_features)
        
    n_values = len(featured_measurements_dict)
    
    fig, ax = plt.subplots(1, n_values, figsize=figsize, sharey=True)
    for i_val, value_name in enumerate(featured_measurements_dict):
        cax = ax[i_val] if n_values > 1 else ax
        cor_mat = np.empty((n_augs, n_augs_aux))
        for i in range(n_augs):
            for j in range(n_augs_aux):
                cor_mat[i, j], _ = scipy.stats.spearmanr(
                    featured_measurements_dict[value_name][i, j, :],
                    featured_measurements_no_aug_input_dict[value_name][j, :],
                    axis=1
                )
        df_vals = pd.DataFrame(
            cor_mat,
            index=augmentation_names,
            columns=augmentation_and_auxilliary_names,
        )
        sns.heatmap(
            df_vals,
            annot=True,
            ax=cax,
            cmap=cmap,
            cbar=False,
            fmt='.3f',
            
        )
        xtick_labels = cax.get_xticklabels()
        if xtick_labels is not None:
            cax.set_xticklabels(xtick_labels, rotation=17.5)
        del df_vals;
        cax.set_xlabel('SA variables used for masking')
        if i_val == 0:
            cax.set_ylabel('Augmentations of input')
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')
        #    cax.set_xticklabels(1. - xticks)
        cax.set_title(f'({value_name})')
    plt.subplots_adjust(wspace=0.025)#, hspace=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
    
import functools
import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import preparation.visualize
import preparation.single_unit

def plot_confusion_matrices(
    confusion_matrices_dict,
    accuracy_dict,
    network_modules,
    augmentation_set_numbers_list,
    values_names,
    n_conv_modules,
    extract_auxilliary_names=True,
    which='valid',
    n_round=3,
    figsize=(30, 18),
    cmap='gist_heat',
    show=True,
    save_dirname=None,
    save_filename_base=None,
):
    assert which in ('train', 'valid')

    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    
    shorten_augmentation_names_dict = dict(
        **preparation.visualize._shorten_augnames_dict,
        **preparation.visualize._shorten_variables_dict
    )

    for i_mn, module_name in enumerate(network_modules):
        i_var = 0
        if i_mn == n_conv_modules:
            break
        #colors = plt.cm.Spectral(np.linspace(0, 1, len(augmentation_names)))
        #colors = plt.cm.plasma_r(np.linspace(0, 1, len(augmentation_names)))
        fig, ax = plt.subplots(
            len(augmentation_set_numbers_list),
            len(values_names),
            figsize=figsize,
        )
        for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
            n_vars = len(augmentation_names_dict[augmentation_set_number])
            current_shorten_names = list(
                map(
                    lambda x: shorten_augmentation_names_dict[x],
                    augmentation_names_dict[augmentation_set_number],
                )
            )
            for i_cval, cur_vals_name in enumerate(values_names):
                acc = accuracy_dict[cur_vals_name][module_name][augmentation_set_number][
                    which
                ]
                df_vals = pd.DataFrame(
                    confusion_matrices_dict[cur_vals_name][module_name][augmentation_set_number][
                        which
                    ].round(n_round),
                    index=current_shorten_names,
                    columns=current_shorten_names,
                )
                sns.heatmap(df_vals, annot=True, ax=ax[i_aug, i_cval], cmap=cmap, cbar=False)
                ax[i_aug, i_cval].set_xticklabels(
                    current_shorten_names,
                    rotation=20
                )
                ax[i_aug, i_cval].set_title(
                    f'({cur_vals_name}) {module_name}, aug.set={augmentation_set_number}; acc={acc:.2f}'
                )
                if i_cval > 0:
                    ax[i_aug, i_cval].set_yticklabels([])
                del df_vals;
            i_var += n_vars
        plt.subplots_adjust(wspace=0.05)#, hspace=0)
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

        
def plot_lda_2d_projections_and_hists(
    projected_sensitivity_values_dict,
    sensitivity_values_dict,
    network_modules,
    augmentation_set_numbers_list,
    values_names,
    n_conv_modules,
    extract_auxilliary_names=True,
    figsize=(30, 8),
    cmap='Spectral',
    show=True,
    alpha=0.5,
):
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )

    for i_mn, module_name in enumerate(network_modules):
        i_var = 0
        #colors = plt.cm.Spectral(np.linspace(0, 1, len(augmentation_names)))
        #colors = plt.cm.plasma_r(np.linspace(0, 1, len(augmentation_names)))
        fig, ax = plt.subplots(len(augmentation_set_numbers_list), len(values_names), figsize=figsize)
        for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
            n_vars = len(augmentation_names_dict[augmentation_set_number])
            colors = getattr(plt.cm, cmap)(np.linspace(0, 1, n_vars), alpha=alpha)
            for i_cval, cur_vals_name in enumerate(values_names):
                if i_mn < n_conv_modules:
                    X = projected_sensitivity_values_dict[cur_vals_name][module_name][augmentation_set_number]
                else:
                    X = sensitivity_values_dict[cur_vals_name][module_name][i_var:i_var+n_vars]
                    n_vars, n_units = X.shape[:2]
                    X = X.reshape((n_vars*n_units, -1))
                n_units = len(X) // n_vars
                y = np.arange(n_vars)[:, None]
                y = np.tile(y, (1, n_units))
                y = y.reshape((n_vars*n_units, ))
                for i in range(n_vars):
                    if i_mn < n_conv_modules:
                        ax[i_aug, i_cval].scatter(
                            X[y == i, 0],
                            X[y == i, 1],
                            color=colors[i],
                            label=augmentation_names_dict[augmentation_set_number][i],
                        )
                    else:
                        ax[i_aug, i_cval].hist(
                            X[y==i],
                            density=True,
                            bins=35,
                            color=colors[i],
                            label=augmentation_names_dict[augmentation_set_number][i]
                        )
                        ax[i_aug, i_cval].set_yticklabels([])
                if i_cval == 0:
                    ax[i_aug, i_cval].legend(
                        loc='center left',
                        bbox_to_anchor=(-0.35, 0.5)
                    )
                    # Shrink current axis by 20%
                    #box = ax.get_position()
                    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                    # Put a legend to the right of the current axis
                    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    #ax[i_aug, i_cval].legend(loc="best", shadow=False, scatterpoints=1, ncol=2)
                ax[i_aug, i_cval].set_title(
                    f'({cur_vals_name}) {module_name}, aug.set={augmentation_set_number}'
                )
            i_var += n_vars
        if show:
            plt.show()
            print('\n\n')

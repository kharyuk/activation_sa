import os
import functools

import numpy as np
#import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.patches
#import matplotlib.gridspec
#import matplotlib.cm
#import h5py

from . import visualize
import preparation.visualize
import preparation.single_unit


def get_global_vmin_vmax(
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list,
    network_module_names,
    values_names_dict,
    n_conv_modules,
    global_bounds_conv_only=True,
    values_funcs=None,
):
    vmin = np.inf
    vmax = -np.inf
    for i_mn, module_name in enumerate(network_module_names):
        if global_bounds_conv_only and (i_mn == n_conv_modules):
            break
        for value_key in values_names_dict:
            for augmentation_set_number in augmentation_set_numbers_list:
                values_path = os.path.join(
                    activations_dirname, values_fnms_dict[value_key][augmentation_set_number]
                )
                for values_name in values_names_dict[value_key]: # select concrete measurement, i.r., means, etc
                    values, act_shape = visualize.get_values(
                        values_path,
                        module_name,
                        values_name,
                        values_func=None if values_funcs is None else values_funcs[value_key][values_name],
                    )
                    values = np.array(values)
                    values = values[np.isfinite(values)]
                    if len(values) == 0:
                        continue
                    vmin = min(values.min(), vmin)
                    vmax = max(values.max(), vmax)
    return vmin, vmax

def get_number_of_neurons(
    values_fnm,
    activations_dirname,
    network_module_names,
    values_name
):
    n_neurons_dict = {}
    for module_name in network_module_names:
        values_path = os.path.join(
            activations_dirname, values_fnm
        )
        values, act_shape = visualize.get_values(
            values_path,
            module_name,
            values_name,
            values_func=None,
        )
        n_neurons_dict[module_name] = int(act_shape[0])
    return n_neurons_dict

##########################################################################
def single_neuron_multiple_values_channeled_plot(
    module_name,
    ij,
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list, # 
    values_names_dict,
    values_funcs=None,
    dataset_part=None,
    ncols_orig=8,
    fig=None,
    figsize=(10, 4),
    gs=None,
    gs_size=None,
    show=True,
    cmap='Reds',
    vmin=0,
    vmax=1,
    plot_colorbar=True
):
    nrows = 1
    ncols = len(augmentation_set_numbers_list)*sum(map(len, values_names_dict.values()))
    #dataset_part = None
    
    if show and plot_colorbar:
        visualize.plot_colorbar(
            figsize=None, w=0.25*figsize[1], h=0.005*figsize[1],
            vmin=vmin, vmax=vmax, dv=0.2*(vmax-vmin), ticks=None,
            label='', cmap=cmap
        )
        plt.show()

    fig, lgs, current_gs = preparation.visualize.init_geometry(
        fig=fig, gs=gs, nrows=nrows, ncols=ncols, figsize=figsize, gs_size=gs_size
    )

    j_col = 0
    i_row = 0
    ij_ind = ij
    
    for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        for i_valkey, value_key in enumerate(values_names_dict):
            values_path = os.path.join(
                activations_dirname, values_fnms_dict[value_key][augmentation_set_number]
            )
            for values_name in values_names_dict[value_key]: # select concrete measurement, i.r., means, etc
                local_values = preparation.visualize.get_single_conv2d_neuron_values(
                    values_fnms_dict,
                    activations_dirname,
                    module_name,
                    ij_ind,
                    ncols_orig,
                    value_key,
                    values_name=values_name,
                    augmentation_set_number=augmentation_set_number,
                    dataset_part=dataset_part,
                    slice_num=0,
                    values_func=None if values_funcs is None else values_funcs[value_key][values_name],
                    shpv_normalize=False,
                )
                ax = plt.subplot(current_gs[i_row, j_col], frameon=False)
                ax = preparation.visualize.ax_disable_ticks_ticklabels(ax)
                ax.imshow(local_values, cmap=cmap, vmin=vmin, vmax=vmax)

                if i_row == 0:
                    xlabel = '\n'
                    if j_col % len(values_names_dict) == 0: #ncols // 2:
                        xlabel = f'Aug.set {augmentation_set_number}'+xlabel

                    ax.set_xlabel(
                        xlabel+f'Scheme {i_valkey+1}',
                        fontsize=10,
                        #rotation=10,
                        loc='center',
                        labelpad=5
                    )
                    ax.xaxis.set_label_position('top')
                    if j_col == 0: #ncols // 2:
                        title = f'{ij}'
                        if isinstance(ij, int):
                            title = f'{preparation.single_unit.ind2pair(ij, ncols_orig)}'
                        ax.set_title(title, x=-0.5, y=1.)

                #if j_col == 0:
                #    rnm = row_name.replace(' ', '\n')
                #    ax.set_ylabel(
                #        f"{rnm}",
                #        fontsize=10,
                #        rotation=69,
                #        loc='center',
                #        labelpad=15
                #    )
                j_col += 1
    
    
    return fig, gs, lgs, current_gs

def multi_neuron_block_plot(
    network_module_names_list,
    plotter_func,
    plotter_func_kwargs_dict,
    colblocks_per_row,
    ind_neurons_dict,
    vmin=0,
    vmax=1,
    cmap='Reds',
    gs_size=None,
    figsize=None,
    plot_colorbar=True,
):
    if plot_colorbar:
        visualize.plot_colorbar(
            figsize=None, w=0.25*figsize[1], h=0.005*figsize[1],
            vmin=0, vmax=vmax, dv=0.2*(vmax-vmin), ticks=None,
            label='', cmap=cmap
        )
        plt.show()

    for module_name in network_module_names_list:
        print('\n\t\t', module_name)
        
        ind_neurons = ind_neurons_dict[module_name]
        if isinstance(ind_neurons, int):
            ind_neurons = list(range(ind_neurons))
        c_nrows = int(np.ceil(len(ind_neurons) / colblocks_per_row))

        fig = plt.figure(constrained_layout=False, figsize=figsize)
        lgs = fig.add_gridspec(
            nrows=c_nrows,
            ncols=colblocks_per_row,
            hspace=0.
        )

        for i, ij in enumerate(ind_neurons):
            if isinstance(ij, int):
                ij_ind = ij
            else:
                ij_ind = preparation.single_unit.pair2ind(ij, ncols_orig)
            plotter_func(
                module_name,
                ij_ind,
                fig=fig,
                #figsize=figsize,
                gs=lgs[i//colblocks_per_row, i%colblocks_per_row],
                gs_size=gs_size or (1, 1),
                #####gs_size=(1, 1),
                show=False,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                plot_colorbar=False,
                **plotter_func_kwargs_dict
            )
        plt.show()
        
def multi_neuron_block_save(
    network_module_names_list,
    plotter_func,
    plotter_func_kwargs_dict,
    ind_neurons_dict,
    save_path,
    save_filename_base,
    vmin=0,
    vmax=1,
    cmap='Reds',
    figsize=None,
    plot_colorbar=True,
    save_fmt='jpg',
):
    os.makedirs(save_path, exist_ok=True)
    fig = visualize.plot_colorbar(
        figsize=None, w=0.25*figsize[1], h=0.005*figsize[1], vmin=0, vmax=vmax,
        dv=0.2*(vmax-vmin), ticks=None, label='', cmap=cmap
    )
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, f'{save_filename_base}_colorbar.{save_fmt}'), bbox_inches='tight')
    plt.clf()
    
    for module_name in network_module_names_list:
        current_fnm = f"{save_filename_base}_{module_name}"
        
        ind_neurons = ind_neurons_dict[module_name]
        if isinstance(ind_neurons, int):
            ind_neurons = list(range(ind_neurons))

        for ij in ind_neurons:
            if isinstance(ij, int):
                ij_ind = ij
            else:
                ij_ind = preparation.single_unit.pair2ind(ij, ncols_orig) # error
            fig, gs, lgs, current_gs = plotter_func(
                module_name,
                ij_ind,
                fig=fig,
                figsize=figsize,
                gs=None,
                gs_size=None,
                show=False,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                plot_colorbar=False,
                **plotter_func_kwargs_dict
            )
            #fig.tight_layout()
            fig.set_size_inches(figsize[0], figsize[1])
            plt.savefig(
                os.path.join(save_path, f'{current_fnm}_{ij_ind}.{save_fmt}'),
                bbox_inches='tight' # needed for means/vars/covs
            )
            plt.clf()
            
    return

def single_neuron_multiple_values_channeled_plot_2(
    module_name,
    ij,
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list, # 
    values_names_dict,
    shpv_group_indices_dict,
    ncols_orig=8,
    fig=None,
    figsize=(10, 4),
    gs=None,
    gs_size=None,
    show=True,
    cmap='Reds',
    vmin=0,
    vmax=1,
    plot_colorbar=True,
):
    row_vals_names = functools.reduce(
        lambda x, y: x+y, values_names_dict.values()
    )
    n_vals = len(row_vals_names)
    max_len_row_vals = max(map(len, row_vals_names))
    
    augmentation_and_auxilliary_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=True,
    )
    augmentation_and_auxilliary_names = functools.reduce(
        lambda x, y: x+y, augmentation_and_auxilliary_names_dict.values()
    )
    n_aug_groups = len(augmentation_set_numbers_list)
    n_aug_aux = len(augmentation_and_auxilliary_names)
    
    if show and plot_colorbar:
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.25*figsize[1], h=0.005*figsize[1],
            vmin=vmin, vmax=vmax, dv=0.2*(vmax-vmin), ticks=None,
            label='', cmap=cmap
        )
        plt.show()
    
    ij_ind = ij
    dataset_part = None
    fig, lgs, current_gs = preparation.visualize.init_geometry(
        fig, gs, n_vals, n_aug_aux, figsize, gs_size
    )
    for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        i_row = 0
        for i_valkey, value_key in enumerate(values_names_dict):
            #values_path = os.path.join(
            #    activations_dirname, values_fnms_dict[value_key][augmentation_set_number]
            #)
            for values_name in values_names_dict[value_key]: # select concrete measurement, i.r., means, etc
                c_values_name = preparation.visualize._values_names_dict[values_name]
                for i_col in range(n_aug_aux):
                    aug_aux_name = augmentation_and_auxilliary_names[i_col]
                    slice_num = i_col
                    for augmentation_set_number in augmentation_set_numbers_list:
                        L = len(augmentation_and_auxilliary_names_dict[augmentation_set_number])
                        if slice_num >= L:
                            slice_num -= L
                        else:
                            break
                    local_values = preparation.visualize.get_single_conv2d_neuron_values(
                        values_fnms_dict,
                        activations_dirname,
                        module_name,
                        ij_ind,
                        ncols_orig,
                        value_key,
                        values_name=c_values_name,
                        augmentation_set_number=augmentation_set_number,
                        dataset_part=dataset_part,
                        slice_num=slice_num,
                        values_func=None if values_name not in preparation.visualize._values_funcs else lambda x: (
                            preparation.visualize._values_funcs[values_name](
                                x,
                                shpv_group_indices_dict[augmentation_set_number]
                            )
                        ),
                        shpv_normalize=True,
                    )
                    ax = plt.subplot(current_gs[i_row, i_col], frameon=False)
                    ax = preparation.visualize.ax_disable_ticks_ticklabels(ax)
                    ax.imshow(local_values, cmap=cmap, vmin=vmin, vmax=vmax)
                    if i_row == 0:
                        if aug_aux_name in preparation.visualize._shorten_variables_dict:
                            xlabel = preparation.visualize._shorten_variables_dict[aug_aux_name]
                        else:
                            xlabel = preparation.visualize._shorten_augnames_dict[aug_aux_name]
                        ax.set_xlabel(
                            xlabel,
                            fontsize=10,
                            rotation=10,
                            loc='center',
                            labelpad=5
                        )
                        ax.xaxis.set_label_position('top')
                        if i_col == 0: #ncols // 2:
                            title = f'{ij}'
                            if isinstance(ij, int):
                                title = f'{preparation.single_unit.ind2pair(ij, ncols_orig)}'
                            ax.set_title(title, x=-0.5, y=1.)

                    if i_col == 0:
                        #rnm = row_name.replace(' ', '\n')
                        rnm = values_name
                        ax.set_ylabel(
                            f"{rnm}",
                            fontsize=10,
                            rotation=0,
                            #loc='center',
                            labelpad=15,
                            verticalalignment='center',
                            horizontalalignment='right'
                        )
                i_row += 1
    loc_pos = current_gs[-1, 0].get_position(fig)
    x0, y0 = loc_pos.x0, loc_pos.y0
    #if gs_flag:
    #    ax = lgs.subplots()[-1, 0]
    #else:
    #    ax = fig.axes[(nrows-1)*ncols]

    gap = 0.125#0.125
    left = loc_pos.x0#*(1.-gap/2) #- loc_pos.width
    bottom = loc_pos.y0#*(1.-gap/2) #+ loc_pos.height
    height = (n_vals+gap)*loc_pos.height

    patches = []

    for i, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        loc_naugs = len(augmentation_and_auxilliary_names_dict[augmentation_set_number])
        width = (loc_naugs + (2.5+1.5*i)*gap)*loc_pos.width

        patches.append(
            matplotlib.patches.Rectangle(#FancyBboxPatch(
                xy=(left, bottom),
                width=width,
                height=height,
                #boxstyle='round',
                fc='none',
                ec=(1., 0.01, 0.2),
                lw=1,
                alpha=0.25
            )
        )
        fig.add_artist(patches[-1])
        left = left + width + loc_pos.width*(gap/2)

    #trans = matplotlib.transforms.blended_transform_factory(ax0.transData, fig.transFigure)
    #r = matplotlib.patches.Rectangle(xy=(xmin,bottom), width=xmax-xmin, height=top-bottom, transform=trans,
    #                                 fc='none', ec='C0', lw=5)
    #fig.add_artist(r)
    if show:
        plt.show()
    return fig, gs, lgs, current_gs
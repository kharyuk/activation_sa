import os
import functools

import matplotlib.patches
import matplotlib.pyplot as plt

import numpy as np

from . import single_unit
import sensitivity_analysis.visualize

#module_name = 'features.5'
#ij = (8, 7)

_row_names = [
    'rbscc (train)',
    'rbscc (valid)',
    'si',
    'siT',
    'shpv'
]

_row_names_cs = [
    'rbscc (train)',
    'rbscc (valid)',    
]

_values_keys_dict = {
    'rbscc (train)': 'cs',
    'rbscc (valid)': 'cs',
    'si': 'si',
    'siT': 'si',
    'shpv': 'shpv'
}

_values_funcs = {
    'cs': lambda x, ind=None: np.abs(x),
    'si': lambda x, ind=None: extract_si_values_func(x, which='si'),
    'siT': lambda x, ind=None: extract_si_values_func(x, which='sT'),
    'shpv': lambda x, ind: extract_shp_values_func(x, ind)
}

_shorten_channels_dict = {
    'hue': 'hue',
    'saturation': 'satur.',
    'value': 'value',
    'grayscale': 'graysc.',
}

_shorten_augnames_dict = {
    'erasing': 'erasing',
    'sharpness_const': 'sharp.',
    'rolling': 'rolling',
    'grayscaling': 'grayscale',
    'gaussian_blur': 'gaus.blur',
    'brightness': 'bright.',
    'contrast': 'contrast',
    'saturation': 'satur.',
    'hue': 'hue',
    'hflip': 'hflip',
    'rotation': 'rotation',
    'elliptic_local_blur': 'el.loc.blur'
}

_shorten_variables_dict = {
    'permutation': 'permut.',
    'class': 'class',
    'partition': 'part.'
}

_values_names_dict = {
    'cs': 'cs_rbscc',
    'si': 'si+sT',
    'shpv': 'shpv',
    'siT': 'si+sT',
}

def extract_shp_values_func(values, group_indices):
    rv = []
    for i in range(len(group_indices)-1):
        ind0, ind1 = group_indices[i:i+2]
        rv.append(np.sum(values[ind0:ind1], axis=0))
    rv = np.array(rv)
    rv = np.clip(rv, 0., None)
    return rv

def extract_si_values_func(values, which):
    possible = ['si', 'sT']
    assert which in possible
    rv = values[possible.index(which)]
    rv = np.clip(rv, 0., 1.)
    return rv

def ax_disable_ticks_ticklabels(ax):
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    return ax

def get_conv2d_unit_values(
    paths_dict,
    results_dirname,
    module_name,
    values_key,
    values_name,
    augmentation_set_number=None,
    dataset_part=None,
    slice_num=None,
    values_func=None,
    shpv_normalize=True
):  
    loc_dict = paths_dict[values_key]
    if values_key == 'cs':
        current_fnm = loc_dict[dataset_part][slice_num]
        values_path = os.path.join(results_dirname, current_fnm)
    else:
        current_fnm = loc_dict[augmentation_set_number]
        values_path = os.path.join(results_dirname, current_fnm)
    
    if values_key != 'shpv':
        values, shape = sensitivity_analysis.visualize.get_values(
            values_path, module_name, values_name, values_func
        )
    else:
        values, shape = sensitivity_analysis.visualize.get_shp_values(
            values_path, module_name, values_name, normalize=shpv_normalize, values_func=values_func
        )
    
    if values_key != 'cs':
        values = values.reshape((-1, )+shape)
        if slice_num is not None:
            values = values[slice_num]
    else:
        values = values.reshape(shape)
    return values


def get_single_conv2d_neuron_values(
    paths_dict,
    results_dirname,
    module_name,
    ij,
    n_cols_orig,
    values_key,
    values_name,
    augmentation_set_number=None,
    dataset_part=None,
    slice_num=None,
    values_func=None,
    shpv_normalize=True
):
    if isinstance(ij, (tuple, list)):
        ij_single = (ij[0]-1)*n_cols_orig+(ij[1]-1)
    elif isinstance(ij, int):
        ij_single = ij
        #ij = ij_single//n_cols_orig+1, ij_single%n_cols_orig+1
    
    values = get_conv2d_unit_values(
        paths_dict,
        results_dirname,
        module_name,
        values_key,
        values_name,
        augmentation_set_number=augmentation_set_number,
        dataset_part=dataset_part,
        slice_num=slice_num,
        values_func=values_func,
        shpv_normalize=shpv_normalize
    )
    #print(values.shape, values_key, values_name)
    return values[ij_single]
    '''
    loc_dict = paths_dict[values_key]
    if values_key == 'cs':
        current_fnm = loc_dict[dataset_part][slice_num]
        values_path = os.path.join(results_dirname, current_fnm)
    else:
        current_fnm = loc_dict[augmentation_set_number]
        values_path = os.path.join(results_dirname, current_fnm)
    
    #print(values_path, ij_single, ij, module_name, values_name, slice_num)
    if values_key != 'shpv':
        values, shape = sensitivity_analysis.visualize.get_values(
            values_path, module_name, values_name, values_func
        )
    else:
        values, shape = sensitivity_analysis.visualize.get_shp_values(
            values_path, module_name, values_name, normalize=True, values_func=values_func
        )
    if values_key != 'cs':
        values = values.reshape((-1, )+shape)
        values = values[slice_num]
    else:
        values = values.reshape(shape)
    ''';

def compute_local_hw(max_h, max_w, n_rows, n_cols):
    #max_hw = max(max_h, max_w)
    #max_w, max_h = max_h / max_hw, max_w / max_hw # not a mistake!
    h_a = max_h / n_rows
    w_a = max_w / n_cols
    a = min(h_a, w_a)
    local_height = a*n_rows / max_h
    local_width = a*n_cols / max_w
    return local_height, local_width


def init_geometry(fig, gs, nrows, ncols, figsize, gs_size):
    #assert not ((fig is None) and (gs is None))
    if gs is not None:
        gs_flag = True
        assert (gs_size is not None) and (fig is not None)
        local_height, local_width = compute_local_hw(gs_size[1], gs_size[0], nrows, ncols)
        lgs = gs.subgridspec(
            nrows=2,
            ncols=2,
            height_ratios=(local_height, 1-local_height),
            width_ratios=(local_width, 1-local_width)
            #wspace=0.25,
            #hspace=0,
        )
    else:
        if fig is None:
            fig = plt.figure(constrained_layout=False, figsize=figsize)
        local_height, local_width = compute_local_hw(figsize[1], figsize[0], nrows, ncols)
        lgs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=(local_height, 1-local_height),
            width_ratios=(local_width, 1-local_width)
        )
    current_gs = lgs[0, 0].subgridspec(nrows=nrows, ncols=ncols, wspace=0.05, hspace=0.05)
    return fig, lgs, current_gs

def single_neuron_aug_plot(
    module_name,
    ij,
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list,
    row_names_list,
    #augmentation_names_dict,
    shpv_group_indices_dict,
    ##row_names,
    #augmentation_names,
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
    if show and plot_colorbar:
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.25*figsize[1], h=0.005*figsize[1],
            vmin=vmin, vmax=vmax, dv=0.2*(vmax-vmin), ticks=None,
            label='', cmap=cmap
        )
    
    assert set(row_names_list).issubset(set(_row_names))
    nrows = len(row_names_list)
    
    augmentation_names_dict = single_unit.extract_augmentation_names_dict(augmentation_set_numbers_list)
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    ncols = len(augmentation_names) ####
    
    gs_flag = False
    fig, lgs, current_gs = init_geometry(fig, gs, nrows, ncols, figsize, gs_size)
    
    dataset_part = None
    #for i_row, row_name in enumerate(row_names):
    for i_row, row_name in enumerate(row_names_list):
        values_key = _values_keys_dict[row_name]
        values_name = _values_names_dict[values_key] # select concrete measurement, i.r., means, etc
        #print(values_key, values_name)
        for i_col in range(ncols):
            aug_name = augmentation_names[i_col]
            for augmentation_set_number in augmentation_set_numbers_list:
                flag = aug_name in augmentation_names_dict[augmentation_set_number]
                if flag:
                    break
            assert flag
            slice_num = i_col
            if values_key != 'cs':
            #if values_name != 'cs':
                L = 0
                for j in augmentation_set_numbers_list:
                    if j == augmentation_set_number:
                        break
                    L += len(augmentation_names_dict[j])
                slice_num -= L
            #print(i_col, slice_num, L, augmentation_set_number)
            if values_key == 'cs':
            #if values_name == 'cs':
                dataset_part = row_name.split(' (')[1].replace(')', '')

            local_values = get_single_conv2d_neuron_values(
                values_fnms_dict,
                activations_dirname,
                module_name,
                ij,
                ncols_orig,
                values_key,
                values_name,#=values_names_dict[values_key],
                augmentation_set_number=augmentation_set_number,
                dataset_part=dataset_part,
                slice_num=slice_num,
                #values_func=None if values_funcs is None else values_funcs[value_key][values_name],
                #shpv_normalize=True,
                values_func=None if row_name not in _values_funcs else lambda x: (
                    _values_funcs[row_name](
                        x,
                        shpv_group_indices_dict[augmentation_set_number]
                    )
                ),
                shpv_normalize=True,
            )
            ax = plt.subplot(current_gs[i_row, i_col], frameon=False)
            #ax.axes.get_xaxis().set_visible(False)
            ax = ax_disable_ticks_ticklabels(ax)
            #ax.axes.get_yaxis().set_visible(False)
            ax.imshow(local_values, cmap=cmap, vmin=vmin, vmax=vmax)
            if i_row == 0:
                ax.set_xlabel(
                    _shorten_augnames_dict[aug_name],
                    fontsize=10,
                    rotation=10,
                    loc='center',
                    labelpad=5
                )
                ax.xaxis.set_label_position('top')
                if i_col == 0:
                    title = f'{ij}'
                    if isinstance(ij, int):
                        title = f'{single_unit.ind2pair(ij, ncols_orig)}'
                    ax.set_title(title, x=-0.5, y=1.)

            if i_col == 0:
                rnm = row_name.replace(' ', '\n')
                ax.set_ylabel(
                    f"{rnm}",
                    fontsize=10,
                    rotation=39,
                    loc='center',
                    labelpad=15
                )
    loc_pos = current_gs[-1, 0].get_position(fig)
    x0, y0 = loc_pos.x0, loc_pos.y0
    #if gs_flag:
    #    ax = lgs.subplots()[-1, 0]
    #else:
    #    ax = fig.axes[(nrows-1)*ncols]

    gap = 0.125
    n_augvals = len( set(row_names_list).difference(set(_row_names_cs)) ) # 3
    left = loc_pos.x0#*(1.-gap/2) #- loc_pos.width
    bottom = loc_pos.y0#*(1.-gap/2) #+ loc_pos.height
    height = (n_augvals+gap)*loc_pos.height

    patches = []

    for i, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        loc_naugs = len(augmentation_names_dict[augmentation_set_number])
        width = (loc_naugs+(1.5+i)*gap)*loc_pos.width

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

def single_neuron_variables_plot(
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list,
    #augmentation_names_dict,
    shpv_group_indices_dict,
    module_name,
    row_names,
    ij,
    variable_names,
    n_cols_orig=8,
    fig=None,
    figsize=(10, 4),
    gs=None,
    gs_size=None,
    show=True,
    cmap='Reds',
    vmin=0,
    vmax=1,
    plot_colorbar=True,
    mark_augset_text=True
):
    if show and plot_colorbar:
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.5*figsize[1], h=0.025*figsize[1], vmin=vmin, vmax=vmax, dv=0.2, ticks=None, label='', cmap=cmap
        )
        
    assert set(row_names).issubset(set(_row_names))
    nrows = len(row_names)
    ncols = len(variable_names) ####
    dataset_part = None
    n_aug_groups = len(augmentation_set_numbers_list)
    
    augmentation_names_dict = single_unit.extract_augmentation_names_dict(augmentation_set_numbers_list)
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )

    
    fig, lgs, current_gs = init_geometry(fig, gs, n_aug_groups*nrows, ncols, figsize, gs_size)

    for i_augset, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        n_aug_var = len(augmentation_names_dict[augmentation_set_number])
        for i_row, row_name in enumerate(row_names):
            values_key = _values_keys_dict[row_name]
            for i_col in range(ncols):
                var_name = variable_names[i_col]
                slice_num = n_aug_var+i_col

                local_values = get_single_conv2d_neuron_values(
                    values_fnms_dict,
                    activations_dirname,
                    module_name,
                    ij,
                    n_cols_orig,
                    values_key,
                    values_name=_values_names_dict[values_key],
                    augmentation_set_number=augmentation_set_number,
                    dataset_part=dataset_part,
                    slice_num=slice_num,
                    values_func=None if row_name not in _values_funcs else lambda x: (
                        _values_funcs[row_name](
                            x,
                            shpv_group_indices_dict[augmentation_set_number]
                        )
                    )
                )
                ax = plt.subplot(current_gs[i_augset*nrows + i_row, i_col], frameon=False)
                #ax.axes.get_xaxis().set_visible(False)
                ax = ax_disable_ticks_ticklabels(ax)
                #ax.axes.get_yaxis().set_visible(False)
                ax.imshow(local_values, cmap=cmap, vmin=vmin, vmax=vmax)
                if (i_row == 0) and (i_augset == 0):
                    ax.set_xlabel(
                        _shorten_variables_dict[var_name],
                        fontsize=10,
                        rotation=10,
                        loc='center',
                        labelpad=5
                    )
                    ax.xaxis.set_label_position('top')
                    if i_col == 0: #ncols // 2:
                        title = f'{ij}'
                        if isinstance(ij, int):
                            title = f'{single_unit.ind2pair(ij, ncols_full)}'
                        ax.set_title(title, x=-0.65, y=1.)
                        
                if mark_augset_text and (i_col == ncols-1) and (i_row % n_aug_groups == nrows // 2):
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(
                        f"augset {augmentation_set_number}",
                        fontsize=10,
                        rotation=90,
                        loc='center',
                        labelpad=15
                    )
                if i_col == 0:
                    rnm = row_name.replace(' ', '\n')
                    ax.set_ylabel(
                        f"{rnm}",
                        fontsize=10,
                        rotation=39,
                        loc='center',
                        labelpad=15
                    )

    loc_pos = current_gs[-1, 0].get_position(fig)
    x0, y0 = loc_pos.x0, loc_pos.y0
    #ax = fig.axes[(n_aug_groups*nrows-1)*ncols]

    gap = 0.125
    n_augvals = len( set(row_names).difference(set(_row_names_cs)) ) # 3
    left = loc_pos.x0#*(1.-gap/2) #- loc_pos.width
    width = (ncols+(1.5+0.5)*gap)*loc_pos.width
    bottom = loc_pos.y0#*(1.-gap/2) #+ loc_pos.height

    patches = []

    for i, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        loc_naugs = len(augmentation_names_dict[augmentation_set_number])
        height = (nrows+gap)*loc_pos.height

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
        #left = left + width + loc_pos.width*(gap/2)
        #ax.text(8, 1, f'augset {augmentation_set_number}', figure=fig)
        #fig.annotate(
        #    patches[-1], (left+width, bottom+height/2), color='w', weight='bold', fontsize=6, ha='center', va='center'
        #)
        bottom = bottom + height
    if show:
        plt.show()
    return fig, gs, lgs, current_gs


def single_neuron_single_values_channeled_plot(
    values_fnm_prefix,
    activations_dirname,
    augmentation_set_number,
    dataset_part,
    config_list,
    #network_module_names,
    values_name,
    values_func=None,
    ncols_max=None,
    ncols_orig=8,
    fig=None,
    figsize=(10, 4),
    #gs=None,
    #gs_size=None,
    show=True,
    cmap='Reds',
    vmin=0,
    vmax=1,
    plot_colorbar=True
):
    fnm_suffix = (
        f'part={dataset_part}'
        f'_augnm={augmentation_set_number}'
    )
    
    network_module_names = list(
        map(
            single_unit.get_network_module_name, config_list
            #lambda x: x['unit_name'], config_list
            #lambda x: f"modules_dict.{x['unit_name']}.{len(x['modules_names'])-1}", config_list
        )
    )
    #print(network_module_names, config_list)
    
    nrows = len(single_unit._HSVG_channel_names)
    
    update_vmin = vmin is None
    update_vmax = vmax is None
    if update_vmin:
        vmin = np.inf
    if update_vmax:
        vmax = -np.inf
    if update_vmin or update_vmax:
        for l, config in enumerate(config_list):
            for (key, neuron_indices) in config['neuron_indices_dict'].items():
                break
            ncols = len(neuron_indices)
            module_name = network_module_names[l]
            for j, channel_name in enumerate(single_unit._HSVG_channel_names):
                tv_fnm = (
                    f'{channel_name}_'
                    f'{values_fnm_prefix}_'
                    f'{fnm_suffix}.hdf5'
                )
                tv_path = os.path.join(activations_dirname, tv_fnm)
                values, val_shape = sensitivity_analysis.visualize.get_values(
                    tv_path, module_name, values_name, values_func
                )
                values = np.array(values)
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                if update_vmin:
                    vmin = min(values.min(), vmin)
                if update_vmax:
                    vmax = max(values.max(), vmax)
    
    
    if show and plot_colorbar:
        #print(vmin, vmax)
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.25*figsize[1], h=0.005*figsize[1], vmin=vmin, vmax=vmax, dv=0.2*(vmax-vmin), ticks=None, label='', cmap=cmap
        )
        plt.show()
    
    
    #for module_name in network_module_names:
    rv_list = []
    
    for l, config in enumerate(config_list):
        for (key, neuron_indices) in config['neuron_indices_dict'].items():
            break
        ncols = len(neuron_indices)
        #title_line1 = f'{config_dict["unit_name"]}\\'
        #print(config['unit_name'], figsize, ncols, nrows)
        module_name = network_module_names[l]
        print(module_name)
        
        fig, lgs, current_gs = init_geometry(
            fig=None, gs=None, nrows=nrows, ncols=ncols_max or ncols, figsize=figsize, gs_size=None
        )
        for j, channel_name in enumerate(single_unit._HSVG_channel_names):
            tv_fnm = (
                f'{channel_name}_'
                f'{values_fnm_prefix}_'
                f'{fnm_suffix}.hdf5'
            )
            tv_path = os.path.join(activations_dirname, tv_fnm)
            values, val_shape = sensitivity_analysis.visualize.get_values(
                tv_path, module_name, values_name, values_func
            )
            values = np.array(values).reshape(val_shape)
            #print(channel_name)
            #fig, ax = plt.subplots(1, n_cols, figsize=(10, 5))
            for i in range(ncols):
                if i < (ncols_max or ncols): # ncols_max if ncols_max is not None else ncols
                    ax = plt.subplot(current_gs[j, i], frameon=False)
                    ax = ax_disable_ticks_ticklabels(ax)
                    #ax.axes.get_xaxis().set_visible(False)
                    #ax.axes.get_yaxis().set_visible(False)
                    ax.imshow(values[i], cmap=cmap, vmin=vmin, vmax=vmax)
                    if j == 0:
                        i_row, i_col = neuron_indices[i]//ncols_orig+1, neuron_indices[i]%ncols_orig+1
                        title = f'({i_row}, {i_col})'
                        ax.set_title(title)
                if i == 0:
                    ax.set_ylabel(
                        f"{_shorten_channels_dict[channel_name]}",
                        fontsize=10,
                        rotation=90,
                        loc='center',
                        labelpad=15
                    )

        if show:
            plt.show()
        rv_list.append((fig, lgs, current_gs))
    return rv_list

def get_global_vmin_vmax(
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list,
    network_module_names,
    dataset_part,
    config_list,
    values_names_dict,
    values_funcs=None,
    slice_num=None,
):
    row_names = list(values_fnms_dict.keys())
    assert set(row_names).issubset(set(single_unit._HSVG_channel_names))
    
    vmin = np.inf
    vmax = -np.inf
    for l, config in enumerate(config_list):
        for (key, neuron_indices) in config['neuron_indices_dict'].items():
            break
        ncols = len(neuron_indices)
        module_name = network_module_names[l]
        for i_row, row_name in enumerate(row_names): # select channel, i.e., hue, value, etc
            current_values_fnm_dict = values_fnms_dict[row_name]
            for value_key in values_names_dict:
                for augmentation_set_number in augmentation_set_numbers_list:
                    for values_name in values_names_dict[value_key]: # select concrete measurement, i.r., means, etc
                        values = get_conv2d_unit_values(
                            current_values_fnm_dict,
                            activations_dirname,
                            module_name,
                            value_key,
                            values_name,
                            augmentation_set_number=augmentation_set_number,
                            dataset_part=dataset_part,
                            slice_num=slice_num,
                            values_func=None if values_funcs is None else values_funcs[value_key][values_name],
                            shpv_normalize=False
                        )
                    
                        values = np.array(values)
                        values = values[np.isfinite(values)]
                        if len(values) == 0:
                            continue
                        vmin = min(values.min(), vmin)
                        vmax = max(values.max(), vmax)
    return vmin, vmax


def single_neuron_multiple_values_channeled_plot(
    neurons_config_dict,
    ij,
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list, # 
    #augmentation_names_dict,
    #shpv_group_indices_dict,
    #module_name, ## another module name!
    #config_list,
    ##values_key,
    values_names_dict,
    # augmentation_names,
    values_funcs=None,
    #ncols_max=None,
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
    row_names = list(values_fnms_dict.keys())
    #assert set(row_names).issubset(set(single_unit._HSVG_channel_names))
    nrows = len(row_names)
    #ncols = len(augmentation_set_numbers_list)*len(values_names_dict)
    ncols = len(augmentation_set_numbers_list)*sum(map(len, values_names_dict.values()))
    #ncols *= len(ij_list)
    
    module_name = single_unit.get_network_module_name(neurons_config_dict)
    dataset_part = None
    
    neuron_indices_list = single_unit.get_neuron_indices_list(neurons_config_dict)
    assert ij in neuron_indices_list
    #assert set(ij_list).issubset(set(neuron_indices_list))
    ij_ind = neuron_indices_list.index(ij)
    
    if show and plot_colorbar:
        #print(vmin, vmax)
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.25*figsize[1], h=0.005*figsize[1], vmin=vmin, vmax=vmax, dv=0.2*(vmax-vmin), ticks=None, label='', cmap=cmap
        )
        plt.show()
    #return

    fig, lgs, current_gs = init_geometry(
        fig=fig, gs=gs, nrows=nrows, ncols=ncols, figsize=figsize, gs_size=gs_size
    )

    for i_row, row_name in enumerate(row_names): # select channel, i.e., hue, value, etc
        current_values_fnm_dict = values_fnms_dict[row_name]
        j_col = 0
        #for value_key in current_values_fnm_dict: # select values, i.e., sitv, shpv, etc
        for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
            for i_valkey, value_key in enumerate(values_names_dict):
                for values_name in values_names_dict[value_key]: # select concrete measurement, i.r., means, etc
                    local_values = get_single_conv2d_neuron_values(
                        current_values_fnm_dict,
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
                    #local_values = local_values[np.isfinite(local_values)]
                    ax = plt.subplot(current_gs[i_row, j_col], frameon=False)
                    ax = ax_disable_ticks_ticklabels(ax)
                    #ax.axes.get_xaxis().set_visible(False)
                    #ax.axes.get_yaxis().set_visible(False)
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
                                title = f'{single_unit.ind2pair(ij, ncols_orig)}'
                            ax.set_title(title, x=-0.5, y=1.)
                    
                    if j_col == 0:
                        rnm = row_name.replace(' ', '\n')
                        ax.set_ylabel(
                            f"{rnm}",
                            fontsize=10,
                            rotation=69,
                            loc='center',
                            labelpad=15
                        )
                    j_col += 1
    
    
    return

def multi_neuron_block_plot(
    analysis_config_list,
    plotter_func,
    plotter_func_kwargs_dict,
    colblocks_per_row,
    N_neurons,
    vmin=0,
    vmax=1,
    cmap='Reds',
    gs_size=None,
    figsize=None,
    plot_colorbar=True,
):
    if plot_colorbar:
        sensitivity_analysis.visualize.plot_colorbar(
            figsize=None, w=0.25*figsize[1], h=0.005*figsize[1], vmin=0, vmax=vmax,
            dv=0.2*(vmax-vmin), ticks=None, label='', cmap=cmap
        )
        plt.show()

    for neurons_config_dict in analysis_config_list:
        print('\n\t\t', neurons_config_dict['unit_name'])
        ijs = list(neurons_config_dict['neuron_indices_dict'].values())[0]
        ijs = ijs[:N_neurons].tolist()

        c_nrows = int(np.ceil(len(ijs) / colblocks_per_row))

        fig = plt.figure(constrained_layout=False, figsize=figsize)
        lgs = fig.add_gridspec(
            nrows=c_nrows,
            ncols=colblocks_per_row,
            hspace=0.
        )

        for i, ij in enumerate(ijs):
            if isinstance(ij, int):
                ij_ind = ij
            else:
                ij_ind = pair2ind(ij, ncols_orig)
            plotter_func(
                neurons_config_dict,
                ij_ind,
                fig=fig,
                figsize=figsize,
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
    analysis_config_list,
    plotter_func,
    plotter_func_kwargs_dict,
    save_path,
    save_filename_base,
    vmin=0,
    vmax=1,
    cmap='Reds',
    figsize=None,
    plot_colorbar=True,
):
    os.makedirs(save_path, exist_ok=True)
    fig = sensitivity_analysis.visualize.plot_colorbar(
        figsize=None, w=0.25*figsize[1], h=0.005*figsize[1], vmin=0, vmax=vmax,
        dv=0.2*(vmax-vmin), ticks=None, label='', cmap=cmap
    )
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, f'{save_filename_base}_colorbar.jpg'), bbox_inches='tight')
    plt.clf()

    for neurons_config_dict in analysis_config_list:
        current_fnm = f"{save_filename_base}_{neurons_config_dict['unit_name']}"
        ijs = list(neurons_config_dict['neuron_indices_dict'].values())[0]
        ijs = ijs.tolist()

        for i, ij in enumerate(ijs):
            if isinstance(ij, int):
                ij_ind = ij
            else:
                ij_ind = pair2ind(ij, ncols_orig)
            plotter_func(
                neurons_config_dict,
                ij_ind,
                fig=None,
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
            plt.savefig(os.path.join(save_path, f'{current_fnm}_{ij_ind}.jpg'))
            plt.clf()

def single_neuron_multiple_values_channeled_plot_2(
    neurons_config_dict,
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
    plot_colorbar=True
):   
    row_chans_names = list(values_fnms_dict.keys())
    assert set(row_chans_names).issubset(set(single_unit._HSVG_channel_names))
    n_chans = len(row_chans_names)
    row_vals_names = functools.reduce(
        lambda x, y: x+y, values_names_dict.values()
    )
    n_vals = len(row_vals_names)
    max_len_row_vals = max(map(len, row_vals_names))
    
    neuron_indices_list = single_unit.get_neuron_indices_list(neurons_config_dict)
    assert ij in neuron_indices_list
    ij_ind = neuron_indices_list.index(ij)
    
    module_name = single_unit.get_network_module_name(neurons_config_dict)
    
    augmentation_and_auxilliary_names_dict = single_unit.extract_augmentation_names_dict(
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
    
    
    fig, lgs, current_gs = init_geometry(
        fig, gs, n_chans*n_vals, n_aug_aux, figsize, gs_size
    )
    dataset_part = None
    for i_block_row, block_row_name in enumerate(row_chans_names): # select channel, i.e., hue, value, etc
        current_values_fnm_dict = values_fnms_dict[block_row_name]
        i_row = i_block_row*n_vals
        for i_valkey, value_key in enumerate(values_names_dict):
            for values_name in values_names_dict[value_key]: # select concrete measurement, i.e., si, etc
                c_values_name = _values_names_dict[values_name]
                for i_col in range(n_aug_aux):
                    aug_aux_name = augmentation_and_auxilliary_names[i_col]
                    slice_num = i_col
                    for augmentation_set_number in augmentation_set_numbers_list:
                        L = len(augmentation_and_auxilliary_names_dict[augmentation_set_number])
                        if slice_num >= L:
                            slice_num -= L
                        else:
                            break
                        
                    local_values = get_single_conv2d_neuron_values(
                        current_values_fnm_dict,
                        activations_dirname,
                        module_name,
                        ij_ind,
                        ncols_orig,
                        value_key,
                        values_name=c_values_name,
                        augmentation_set_number=augmentation_set_number,
                        dataset_part=dataset_part,
                        slice_num=slice_num,
                        values_func=None if values_name not in _values_funcs else lambda x: (
                            _values_funcs[values_name](
                                x,
                                shpv_group_indices_dict[augmentation_set_number]
                            )
                        ),
                        shpv_normalize=True,
                    )
                    ax = plt.subplot(current_gs[i_row, i_col], frameon=False)
                    ax = ax_disable_ticks_ticklabels(ax)
                    ax.imshow(local_values, cmap=cmap, vmin=vmin, vmax=vmax)
                    if i_row == 0:
                        if aug_aux_name in _shorten_variables_dict:
                            xlabel = _shorten_variables_dict[aug_aux_name]
                        else:
                            xlabel = _shorten_augnames_dict[aug_aux_name]
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
                                title = f'{single_unit.ind2pair(ij, ncols_orig)}'
                            ax.set_title(title, x=-0.5, y=1.)

                    if i_col == 0:
                        rnm = ' '
                        if i_row % n_vals == 0:
                            rnm = _shorten_channels_dict[block_row_name] + ': ' + rnm*(1 + max_len_row_vals - len(values_name))
                        #rnm = row_name.replace(' ', '\n')
                        rnm += values_name
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

    gap = 3.5*0.125#0.125
    n_augvals = n_chans*n_vals # 3
    left = loc_pos.x0#*(1.-gap/2) #- loc_pos.width
    bottom = loc_pos.y0#*(1.-gap/2) #+ loc_pos.height
    height = (n_augvals+gap)*loc_pos.height

    patches = []

    for i, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        loc_naugs = len(augmentation_and_auxilliary_names_dict[augmentation_set_number])
        width = (loc_naugs + (0.8+0.25*i)*gap)*loc_pos.width

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
        left = left + width + loc_pos.width*(gap/8)

    #trans = matplotlib.transforms.blended_transform_factory(ax0.transData, fig.transFigure)
    #r = matplotlib.patches.Rectangle(xy=(xmin,bottom), width=xmax-xmin, height=top-bottom, transform=trans,
    #                                 fc='none', ec='C0', lw=5)
    #fig.add_artist(r)
    if show:
        plt.show()
    return fig, gs, lgs, current_gs

# how to localize:
# - select value [cs, si, siT, shpv]
# - select module
# - select neuron

def ind2pair(ind, ncols_full):
    i, j = ind//ncols_full+1, ind%ncols_full+1
    return i, j

def pair2ind(ij, ncols_full):
    return (ij[0]-1)*ncols_full + (ij[1]-1)


def single_neuron_aug_channeled_fixed_value_plot(
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list, # 
    #augmentation_names_dict,
    shpv_group_indices_dict,
    #module_name, ## another module name!
    neurons_config_dict,
    values_key,
    ij, ## preselected navigation !
    # augmentation_names,
    n_cols_orig=8,
    fig=None,
    figsize=(10, 4),
    gs=None,
    gs_size=None,
    show=True,
    cmap='Reds',
    vmin=0,
    vmax=1
):
    augmentation_names_dict = single_unit.extract_augmentation_names_dict(augmentation_set_numbers_list)
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    
    row_names = list(values_fnms_dict.keys())
    assert set(row_names).issubset(set(single_unit._HSVG_channel_names))
    nrows = len(row_names)
    ncols = len(augmentation_names) ####
    
    module_name = single_unit.get_network_module_name(neurons_config_dict)
    
    neuron_indices_list = single_unit.get_neuron_indices_list(neurons_config_dict)
    assert ij in neuron_indices_list
    ij_ind = neuron_indices_list.index(ij)
    
    fig, lgs, current_gs = init_geometry(fig, gs, nrows, ncols, figsize, gs_size)
    
    dataset_part = None
    for i_row, row_name in enumerate(row_names):
        current_values_fnm_dict = values_fnms_dict[row_name]
        for i_col in range(ncols):
            aug_name = augmentation_names[i_col]
            for augmentation_set_number in augmentation_set_numbers_list:
                flag = aug_name in augmentation_names_dict[augmentation_set_number]
                if flag:
                    break
            assert flag
            slice_num = i_col
            if values_key != 'cs':
                L = 0
                for j in augmentation_set_numbers_list:
                    if j == augmentation_set_number:
                        break
                    L += len(augmentation_names_dict[j])
                slice_num -= L
            #print(i_col, slice_num, L, augmentation_set_number)
            if values_key == 'cs':
                dataset_part = row_name.split(' (')[1].replace(')', '')
            local_values = get_single_conv2d_neuron_values(
                current_values_fnm_dict,
                activations_dirname,
                module_name,
                ij_ind,
                n_cols_orig,
                values_key,
                values_name=_values_names_dict[values_key],
                augmentation_set_number=augmentation_set_number,
                dataset_part=dataset_part,
                slice_num=slice_num,
                values_func=None if row_name not in _values_funcs else lambda x: (
                    _values_funcs[row_name](
                        x,
                        shpv_group_indices_dict[augmentation_set_number]
                    )
                )
            )
            ax = plt.subplot(current_gs[i_row, i_col], frameon=False)
            #ax.axes.get_xaxis().set_visible(False)
            ax = ax_disable_ticks_ticklabels(ax)
            #ax.axes.get_yaxis().set_visible(False)
            ax.imshow(local_values, cmap=cmap, vmin=vmin, vmax=vmax)
            if i_row == 0:
                ax.set_xlabel(
                    _shorten_augnames_dict[aug_name],
                    fontsize=10,
                    rotation=10,
                    loc='center',
                    labelpad=5
                )
                ax.xaxis.set_label_position('top')
                if i_col == 0: #ncols // 2:
                    title = f'{ij}'
                    if isinstance(ij, int):
                        title = f'{single_unit.ind2pair(ij, n_cols_orig)}'
                    ax.set_title(title, x=-0.5, y=1.)
                
                
            if i_col == 0:
                rnm = row_name.replace(' ', '\n')
                ax.set_ylabel(
                    f"{rnm}",
                    fontsize=10,
                    rotation=39,
                    loc='center',
                    labelpad=15
                )
    loc_pos = current_gs[-1, 0].get_position(fig)
    x0, y0 = loc_pos.x0, loc_pos.y0
    #if gs_flag:
    #    ax = lgs.subplots()[-1, 0]
    #else:
    #    ax = fig.axes[(nrows-1)*ncols]

    gap = 0.125
    n_augvals = len(row_names) # 3
    left = loc_pos.x0#*(1.-gap/2) #- loc_pos.width
    bottom = loc_pos.y0#*(1.-gap/2) #+ loc_pos.height
    height = (n_augvals+gap)*loc_pos.height

    patches = []

    for i, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        loc_naugs = len(augmentation_names_dict[augmentation_set_number])
        width = (loc_naugs+(1.5+i)*gap)*loc_pos.width

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

def single_neuron_aug_value_fixed_channel_plot(
    values_fnms_dict,
    activations_dirname,
    augmentation_set_numbers_list, # 
    #augmentation_names_dict,
    shpv_group_indices_dict,
    #module_name, ## another module name!
    neurons_config_dict,
    row_names,
    values_key,
    ij, ## preselected navigation !
    # augmentation_names,
    n_cols_orig=8,
    fig=None,
    figsize=(10, 4),
    gs=None,
    gs_size=None,
    show=True,
    cmap='Reds',
    vmin=0,
    vmax=1
):
    
    assert set(row_names).issubset(set(_row_names))
    nrows = len(row_names)
    
    augmentation_names_dict = single_unit.extract_augmentation_names_dict(augmentation_set_numbers_list)
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    
    ncols = len(augmentation_names) ####
    dataset_part = None
    n_aug_groups = len(augmentation_set_numbers_list)
    
    module_name = single_unit.get_network_module_name(neurons_config_dict)
    
    neuron_indices_list = single_unit.get_neuron_indices_list(neurons_config_dict)
    assert ij in neuron_indices_list
    ij_ind = neuron_indices_list.index(ij)
    
    fig, lgs, current_gs = init_geometry(fig, gs, nrows, ncols, figsize, gs_size)
    
    dataset_part = None
    for i_row, row_name in enumerate(row_names): # value like cs / ...
        #current_values_fnm_dict = values_fnms_dict[row_name]
        for i_col in range(ncols):
            aug_name = augmentation_names[i_col]
            for augmentation_set_number in augmentation_set_numbers_list:
                flag = aug_name in augmentation_names_dict[augmentation_set_number]
                if flag:
                    break
            assert flag
            slice_num = i_col
            if values_key != 'cs':
                L = 0
                for j in augmentation_set_numbers_list:
                    if j == augmentation_set_number:
                        break
                    L += len(augmentation_names_dict[j])
                slice_num -= L
            #print(i_col, slice_num, L, augmentation_set_number)
            if values_key == 'cs':
                dataset_part = row_name.split(' (')[1].replace(')', '')
            local_values = get_single_conv2d_neuron_values(
                values_fnms_dict,
                activations_dirname,
                module_name,
                ij_ind,
                n_cols_orig,
                values_key,
                values_name=_values_names_dict[values_key],
                augmentation_set_number=augmentation_set_number,
                dataset_part=dataset_part,
                slice_num=slice_num,
                values_func=None if row_name not in _values_funcs else lambda x: (
                    _values_funcs[row_name](
                        x,
                        shpv_group_indices_dict[augmentation_set_number]
                    )
                )
            )
            ax = plt.subplot(current_gs[i_row, i_col], frameon=False)
            #ax.axes.get_xaxis().set_visible(False)
            ax = ax_disable_ticks_ticklabels(ax)
            #ax.axes.get_yaxis().set_visible(False)
            ax.imshow(local_values, cmap=cmap, vmin=vmin, vmax=vmax)
            if i_row == 0:
                ax.set_xlabel(
                    _shorten_augnames_dict[aug_name],
                    fontsize=10,
                    rotation=10,
                    loc='center',
                    labelpad=5
                )
                ax.xaxis.set_label_position('top')
                if i_col == 0: #ncols // 2:
                    title = f'{ij}'
                    if isinstance(ij, int):
                        title = f'{single_unit.ind2pair(ij, n_cols_orig)}'
                    ax.set_title(title, x=-0.65, y=1.)
            if i_col == 0:
                rnm = row_name.replace(' ', '\n')
                ax.set_ylabel(
                    f"{rnm}",
                    fontsize=10,
                    rotation=39,
                    loc='center',
                    labelpad=15
                )
    loc_pos = current_gs[-1, 0].get_position(fig)
    x0, y0 = loc_pos.x0, loc_pos.y0
    #if gs_flag:
    #    ax = lgs.subplots()[-1, 0]
    #else:
    #    ax = fig.axes[(nrows-1)*ncols]

    gap = 0.125
    n_augvals = len( set(row_names).difference(set(_row_names_cs)) ) # 3
    left = loc_pos.x0#*(1.-gap/2) #- loc_pos.width
    bottom = loc_pos.y0#*(1.-gap/2) #+ loc_pos.height
    height = (n_augvals+gap)*loc_pos.height

    patches = []

    for i, augmentation_set_number in enumerate(augmentation_set_numbers_list):
        loc_naugs = len(augmentation_names_dict[augmentation_set_number])
        width = (loc_naugs+(1.5+i)*gap)*loc_pos.width

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



'''
def single_neuron_aug_channeled_fixed_vlue_plot(
    values_fnm_prefix,
    activations_dirname,
    augmentation_set_number,
    dataset_part,
    config_list,
    #network_module_names,
    values_name,
    values_func=None,
    ncols_max=None,
    ncols_orig=8,
    fig=None,
    figsize=(10, 4),
    #gs=None,
    #gs_size=None,
    show=True,
    cmap='Reds'
):
    fnm_suffix = (
        f'part={dataset_part}'
        f'_augnm={augmentation_set_number}'
    )
    
    network_module_names = list(
        map(
            #lambda x: x['unit_name'], config_list
            lambda x: f"modules_dict.{x['unit_name']}.{len(x['modules_names'])-1}",
            config_list
        )
    )
    
    nrows = len(single_unit._HSVG_channel_names)
    
    #for module_name in network_module_names:
    rv_list = []
    
    for l, config in enumerate(config_list):
        for (key, neuron_indices) in config['neuron_indices_dict'].items():
            break
        ncols = len(neuron_indices)
        #title_line1 = f'{config_dict["unit_name"]}\\'
        print(config['unit_name'], figsize, ncols, nrows)
        module_name = network_module_names[l]
        
        fig, lgs, current_gs = init_geometry(
            fig=None, gs=None, nrows=nrows, ncols=ncols_max or ncols, figsize=figsize, gs_size=None
        )
        for j, channel_name in enumerate(single_unit._HSVG_channel_names):
            tv_fnm = (
                f'{channel_name}_'
                f'{values_fnm_prefix}_'
                f'{fnm_suffix}.hdf5'
            )
            tv_path = os.path.join(activations_dirname, tv_fnm)
            values, val_shape = sensitivity_analysis.visualize.get_values(
                tv_path, module_name, values_name, values_func
            )
            values = np.array(values).reshape(val_shape)
            #print(channel_name)
            #fig, ax = plt.subplots(1, n_cols, figsize=(10, 5))
            for i in range(ncols):
                if i < (ncols_max or ncols): # ncols_max if ncols_max is not None else ncols
                    ax = plt.subplot(current_gs[j, i], frameon=False)
                    ax = ax_disable_ticks_ticklabels(ax)
                    #ax.axes.get_xaxis().set_visible(False)
                    #ax.axes.get_yaxis().set_visible(False)
                    ax.imshow(values[i], cmap=cmap)
                    if j == 0:
                        i_row, i_col = neuron_indices[i]//ncols_orig+1, neuron_indices[i]%ncols_orig+1
                        title = f'({i_row}, {i_col})'
                        ax.set_title(title)
                if i == 0:
                    ax.set_ylabel(
                        f"{_shorten_channels_dict[channel_name]}",
                        fontsize=10,
                        rotation=90,
                        loc='center',
                        labelpad=15
                    )

        if show:
            plt.show()
        rv_list.append((fig, lgs, current_gs))
    return rv_list

'''
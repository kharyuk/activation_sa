import os

import numpy as np
#import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.ticker
#import matplotlib.cm

#import h5py

#import sensitivity_analysis.shapley
import sensitivity_analysis.visualize

def reshape_values(values, act_shape):
    return np.reshape(values, (-1,) + act_shape)

def reconstruct_conv_projected_image(
    values, image_size=224, patch_size=16, in_channels=3
):
    batch_dim, hidden_dim, feature_dim = values.shape
    hidden_dim -= 1
    n_h = image_size // patch_size
    n_w = image_size // patch_size
    assert hidden_dim == n_h * n_w
    assert feature_dim == patch_size*patch_size*in_channels
    result = np.reshape(
        values[:, 1:, :], (batch_dim, n_h, n_w, patch_size, patch_size, in_channels)
    )
    result = np.transpose(result, (0, 5, 1, 3, 2, 4))
    return np.reshape(result, (batch_dim, in_channels, image_size, image_size))

def get_common_patch(values):
    batch_dim, hidden_dim, feature_dim = values.shape
    return values[:, 0, :]

def tfb_post_values_func(values, act_shape):
    if len(act_shape) == 1:
        return values, act_shape
    values = np.reshape(values, (-1, ) + act_shape)
    values = np.transpose(values, (0, 3, 1, 2))
    return values, values.shape[1:]

def custom_plot_single_value_tfb_images(
    values_path,
    network_module_names,
    values_name,
    image_size=223,
    patch_size=16,
    n_channels=3,
    vmin=None,
    vmax=None,
    values_func=None,
    post_values_func=None,
    save_path=None,
    cmap_name="Reds",
    figsize=None,
    show_colorbar=True,
    global_colorbounds=False,
    colorbar_path=None,
    show=True,
    text_size="large",
    
):
    n_modules = len(network_module_names)
    if global_colorbounds:
        if vmin is None:
            vmin = np.inf # vmin = vmin or np.inf, but sometimes better write it explicitly
        if vmax is None:
            vmax = -np.inf
        for i_mn, module_name in enumerate(network_module_names):
            values, act_shape = sensitivity_analysis.visualize.get_values(
                values_path, module_name, values_name, values_func
            )
            values = np.reshape(values, (-1,) + act_shape)
            values = reconstruct_conv_projected_image(
                values, image_size, patch_size, n_channels
            )
            vmin = min(vmin, values.min())
            vmax = max(vmax, values.max())
    
    offset = 1
    n_rows_list = []
    values = []
    for i_mn, module_name in enumerate(network_module_names):
        current_values, act_shape = sensitivity_analysis.visualize.get_values(
            values_path, module_name, values_name, values_func
        )
        current_values = np.reshape(current_values, (-1,) + act_shape)
        current_values = reconstruct_conv_projected_image(
            current_values, image_size, patch_size, n_channels
        )
        current_values = np.reshape(current_values, (n_channels*image_size, image_size))
        values.append(current_values)
        n_rows_list.append(current_values.shape[0])
    max_n_rows = max(n_rows_list)
    if post_values_func is not None:
        values = post_values_func(values)
    
    if show_colorbar:
        assert (vmin is not None) and (vmax is not None)
        cb_fig = sensitivity_analysis.visualize.plot_colorbar(
            figsize=(30, 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax, dv=(vmax-vmin)/5, ticks=None,
            label='', cmap=cmap_name
        )
        if colorbar_path is not None:
            plt.savefig(colorbar_path, bbox_inches='tight')
        if show:
            plt.show(cb_fig)
        else:
            plt.clf()

    fig, ax = plt.subplots(1, n_modules, figsize=figsize)
    for i_module in range(n_modules):
        cax = ax[i_module]
        cax.axis("off")
        
        coef = 1/n_rows_list[i_module]
        delta_y = 0.5*coef
        cax.text(0.05, 1+delta_y, f"{network_module_names[i_module]}\n", size=text_size)
        
        pcm = matplotlib.cm.ScalarMappable(
            matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
            cmap=cmap_name
        )
        delta_y = 0.06 * coef
        cax.matshow(
            values[i_module],
            interpolation='none',
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax
        )
        
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.clf()
    return fig
    # figsize=((n_cols*n_col_blocks)//2, n_rows),

def basic_plot_fc(
    ax,
    images,
    masks=None,
    offset=1,
    vmin=None,
    vmax=None,
    color_name='red',
    mask_color_name='firebrick',
    disable_yticks=False,
    disable_margin_spaces=True
):
    t = np.arange(images.size)
    if masks is not None:
        width = 0.25
        mask_im = ax.barh(
            t, masks.flatten(), height=width, color=mask_color_name, alpha=0.5
        )
        im = ax.barh(
            t+width, images.flatten(), height=2*width, color=color_name,
        )
    else:
        im = ax.barh(
            t, images.flatten(), color=color_name,
        )
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
    ax.set_xlim(
        left=images.min() if vmin is None else vmin,
        right=images.max() if vmax is None else vmax
    )
    ax.invert_yaxis()
    ax.grid(alpha=0.5)
    xticklabels = ax.axes.get_xticklabels()
    for i, xlabel in enumerate(xticklabels):
        if 0 < i < len(xticklabels)-1:
            continue
        xlabel.set_visible(False)
        xlabel.set_fontsize(0.)
    
    
    if disable_yticks:
        ax.axes.get_yaxis().set_visible(False)
    if disable_margin_spaces:
        ax.axes.margins(x=0, y=0)
    
    return im

def custom_plot_single_value_tfb_plots(
    values_path,
    network_module_names,
    values_name,
    n_non_fc_modules,
    image_size=223,
    patch_size=16,
    n_channels=3,
    vmin=None,
    vmax=None,
    values_func=None,
    post_values_func=None,
    save_path=None,
    color_name="red",
    figsize=None,
    global_colorbounds=False,
    show=True,
    text_size="large",
):
    n_modules = len(network_module_names)
    if global_colorbounds:
        if vmin is None:
            vmin = np.inf # vmin = vmin or np.inf, but sometimes better write it explicitly
        if vmax is None:
            vmax = -np.inf
        for i_mn, module_name in enumerate(network_module_names):
            values, act_shape = sensitivity_analysis.visualize.get_values(
                values_path, module_name, values_name, values_func
            )
            if i_mn < n_non_fc_modules:
                values = np.reshape(values, (-1,) + act_shape)
                values = get_common_patch(values)
            vmin = min(vmin, values.min())
            vmax = max(vmax, values.max())
    
    offset = 1
    n_rows_list = []
    values = []
    for i_mn, module_name in enumerate(network_module_names):
        current_values, act_shape = sensitivity_analysis.visualize.get_values(
            values_path, module_name, values_name, values_func
        )
        if i_mn < n_non_fc_modules:
            current_values = np.reshape(current_values, (-1,) + act_shape)
            current_values = get_common_patch(current_values)[0]
        values.append(current_values)
        n_rows_list.append(current_values.shape[0])
    if post_values_func is not None:
        values = post_values_func(values)
    fig, ax = plt.subplots(1, n_modules, figsize=figsize)
    for i_module in range(n_modules):
        if n_modules == 1:
            cax = ax
        else:
            cax = ax[i_module]
        #cax.axis("off")
        
        coef = 1/n_rows_list[i_module]
        delta_y = 0.5*coef
        #cax.text(0.05, 1+delta_y, f"{network_module_names[i_module]}\n", size=text_size)
        cax.set_title(f"{network_module_names[i_module]}\n", size=text_size)
        #delta_y = 0.06 * coef

        basic_plot_fc(
            cax,
            values[i_module],
            #offset=offset,
            vmin=vmin,
            vmax=vmax,
            color_name=color_name,
            disable_yticks=False,
            disable_margin_spaces=True
        )
        #cax.axes.get_xaxis().set_visible(False)
        #cax.axes.get_yaxis().set_visible(False)

        
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.clf()
    return fig

def plot_decomposed_images(
    images,
    variable_names,
    masks=None,
    offset=1,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    block_names=None,
    cmap_name='Reds',
    mask_cmap_name='Reds',
    show_colorbar=True,
    colorbar_path=None,
    text_size='large'
):
    nvars = len(variable_names)
    nblocks = len(images)
    n_rows = images.shape[0]
    n_cols = images.shape[1]
        
    if show_colorbar:
        assert (vmin is not None) and (vmax is not None)
        cb_fig = sensitivity_analysis.visualize.plot_colorbar(
            figsize=(figsize[0], 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax,
            dv=(vmax-vmin)/5, ticks=None, label='', cmap=cmap_name
        )
        #plt.tight_layout()
        if colorbar_path is not None:
            plt.savefig(colorbar_path, bbox_inches='tight')
        if show:
            plt.show(cb_fig)
        else:
            plt.clf()
    
    coef_y = 0.1*n_cols
    coef_x = 0.1*n_rows
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        nrows=nblocks, ncols=nvars, wspace=0.01, hspace=0.01,
    )
    #inner_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
    #    n_rows, n_cols, gs, wspace=0, hspace=0
    #)
    current_mask = None
    for i in range(nblocks):
        ax = plt.subplot(gs[i, 0], frameon=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if block_names is not None:
            ax.text(-0.15, 0.5, block_names[i], size=text_size)
        for j in range(nvars):
            local_gs = gs[i, j]
            if i == 0:
                ax = plt.subplot(local_gs, frameon=False)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                #ax.text(0.05, 1+coef_y/n_rows, f'{variable_names[j]}', size=text_size)
                ax.set_title(f'{variable_names[j]}', size=text_size)
            
            if masks is not None:
                current_mask = masks[i, j]

            ax = fig.add_subplot(local_gs)
            if masks is not None:
                mask_im = ax.imshow(bg_array)
                current_mask = masks[ind][0]
            im = ax.matshow(
                images[i, j],
                alpha=current_mask,
                cmap=cmap_name,
                interpolation='none',
                vmin=vmin,
                vmax=vmax,
            )                
            ax.axis("off")
    #plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
    return fig
            

def si_compact_plot(
    si_path,
    network_module_names,
    values_name,
    variable_names,
    row_names,
    image_size=224,
    patch_size=16,
    n_channels=3,
    vmin=0,
    vmax=1,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    cmap_name='Reds',
    show=True,
    show_colorbar=True,
    figsize=(5, 3)
):
    save_path = None
    colorbar_path = None
    n_rows = len(row_names)
    n_modules = len(network_module_names)
    n_cols = len(variable_names)
    for i_mn, module_name in enumerate(network_module_names):
        values, act_shape = sensitivity_analysis.visualize.get_values(
            si_path, module_name, values_name, values_func
        )
        values = np.reshape(values, (-1, ) + act_shape)
        values = reconstruct_conv_projected_image(
            values, image_size, patch_size, n_channels
        )
        values = np.reshape(
            values, (n_rows, n_cols, n_channels*image_size, image_size)
        )
        print(module_name)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            colorbar_filename = f'{save_filename_base}_{module_name}_colorbar.pdf'
            colorbar_path = os.path.join(save_dirname, colorbar_filename)
        
        plot_decomposed_images(
            images=values,
            variable_names=variable_names,
            figsize=figsize,
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            block_names=row_names,
            cmap_name=cmap_name,
            show_colorbar=show_colorbar,
            colorbar_path=colorbar_path
        )

def si_compact_plot_fc(
    si_path,
    network_module_names,
    n_non_fc_modules,
    values_name,
    variable_names,
    row_names,
    image_size=224,
    patch_size=16,
    n_channels=3,
    vmin=0,
    vmax=1,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    color_name="red",
    show=True,
    figsize=(5, 3),
    text_size="large"
):
    save_path = None
    colorbar_path = None
    n_rows = len(row_names)
    n_modules = len(network_module_names)
    n_cols = len(variable_names)
    for i_mn, module_name in enumerate(network_module_names):
        values, act_shape = sensitivity_analysis.visualize.get_values(
            si_path, module_name, values_name, values_func
        )
        if i_mn < n_non_fc_modules:
            values = np.reshape(values, (-1,) + act_shape)
            values = get_common_patch(values)
        values = np.reshape(
            values, (n_rows, n_cols, -1)
        )
        print(module_name)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        for i_block in range(n_rows):
            for i_var in range(n_cols):
                cax = ax[i_block, i_var]
                if i_block == 0:
                    cax.set_title(f"{variable_names[i_var]}", size=text_size)
                if i_var == 0:
                    cax.set_ylabel(f"{row_names[i_block]}    ", rotation=0)
                disable_yticks = i_var > 0
                basic_plot_fc(
                    cax,
                    values[i_block, i_var],
                    #offset=offset,
                    vmin=vmin,
                    vmax=vmax,
                    color_name=color_name,
                    disable_yticks=disable_yticks,
                    disable_margin_spaces=True
                )
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()

def plot_si2_images(
    images,
    variable_names,
    n_rows,
    n_cols,
    masks=None,
    offset=1,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    cmap_name='Reds',
    mask_cmap_name='Reds',
    show_colorbar=True,
    text_size='large'
):
    batch_size = images.shape[0]
    offset = batch_size // (n_rows*n_cols)
    offset = max(1, offset)
    if figsize is None:
        figsize = (1.5*n_cols, 1.5*n_rows)
    nvars = len(variable_names)
    
    if show_colorbar:
        assert (vmin is not None) and (vmax is not None) and (show)
        cb_fig = sensitivity_analysis.visualize.plot_colorbar(
            figsize=(figsize[0], 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax,
            dv=(vmax-vmin)/5, ticks=None, label='', cmap=cmap_name
        )
        plt.show(cb_fig)
    
    height_ratios = np.ones(n_rows+2)
    width_ratios = np.ones(n_cols+2)
    coef_y = 0.05*n_cols
    coef_x = 0.05*n_rows
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        nrows=nvars, ncols=nvars, wspace=0.1, hspace=0.1,
    )
    
    current_mask = None
    n_col_blocks = n_row_blocks = nvars
    ind = 0
    for i in range(nvars):
        for j in range(nvars):
            local_gs = gs[i, j]
            local_max_width = figsize[0] / n_col_blocks
            local_max_height = figsize[1] / n_row_blocks
            h_a = local_max_height / n_rows
            w_a = local_max_width / n_cols
            a = min(h_a, w_a)
            local_height = a*n_rows / local_max_height
            local_width = a*n_cols / local_max_width
            
            sgc = local_gs.subgridspec(
                nrows=1,
                ncols=1,
            )
            local_subsgc = sgc[0, 0]
            if i >= j:
                continue
            ax = plt.subplot(local_subsgc, frameon=False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            coef = n_cols / n_rows
            delta_y = 0.05*coef
            ax.set_title(f'{variable_names[i]}--{variable_names[j]}', size=text_size)
            if masks is not None:
                current_mask = masks[ind]
            ax.matshow(
                images[ind],
                alpha=current_mask,
                cmap=cmap_name,
                interpolation='none',
                vmin=vmin,
                vmax=vmax,
            ) 
            ax.axis("off")
            ind += 1
    
    fig.subplots_adjust(wspace=0., hspace=0.)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig

def si2_compact_plot(
    si_path,
    network_module_names,
    values_name,
    variable_names,
    image_size=224,
    patch_size=16,
    n_channels=3,
    vmin=0,
    vmax=1,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    cmap_name='Reds',
    show=True,
    show_colorbar=True,
    figsize=(5, 3)
):
    save_path = None
    colorbar_path = None
    #n_rows = len(row_names)
    n_modules = len(network_module_names)
    n_cols = len(variable_names)
    n_rows = n_cols
    for i_mn, module_name in enumerate(network_module_names):
        values, act_shape = sensitivity_analysis.visualize.get_values(
            si_path, module_name, values_name, values_func
        )
        values = np.reshape(values, (-1, ) + act_shape)
        values = reconstruct_conv_projected_image(
            values, image_size, patch_size, n_channels
        )
        values = np.reshape(
            values, (-1, n_channels*image_size, image_size)
        )
        print(module_name)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            colorbar_filename = f'{save_filename_base}_{module_name}_colorbar.pdf'
            colorbar_path = os.path.join(save_dirname, colorbar_filename)
        
        plot_si2_images(
            images=values,
            n_rows=n_rows,
            n_cols=n_cols,
            variable_names=variable_names,
            figsize=figsize,
            #figsize=(0.65*n_cols, 0.65*n_rows),
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            cmap_name=cmap_name,
            show_colorbar=show_colorbar
        )

def plot_si2_fc(
    values,
    variable_names,
    n_rows,
    n_cols,
    masks=None,
    offset=1,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    color_name='red',
    mask_color_name='firebrick',
    text_size='large'
):
    batch_size = values.shape[0]
    offset = batch_size // (n_rows*n_cols)
    offset = max(1, offset)
    if figsize is None:
        figsize = (1.5*n_cols, 1.5*n_rows)
    nvars = len(variable_names)
    
    height_ratios = np.ones(n_rows+2)
    width_ratios = np.ones(n_cols+2)
    coef_y = 0.05*n_cols
    coef_x = 0.05*n_rows
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        nrows=nvars, ncols=nvars, wspace=0.1, hspace=0.1,
    )
    
    current_mask = None
    n_col_blocks = n_row_blocks = nvars
    ind = 0
    for i in range(nvars):
        for j in range(nvars):
            local_gs = gs[i, j]
            local_max_width = figsize[0] / n_col_blocks
            local_max_height = figsize[1] / n_row_blocks
            h_a = local_max_height / n_rows
            w_a = local_max_width / n_cols
            a = min(h_a, w_a)
            local_height = a*n_rows / local_max_height
            local_width = a*n_cols / local_max_width
            
            sgc = local_gs.subgridspec(
                nrows=1,
                ncols=1,
            )
            local_subsgc = sgc[0, 0]
            if i >= j:
                continue
            ax = plt.subplot(local_subsgc, frameon=True)
            #ax.axes.get_xaxis().set_visible(False)
            #ax.axes.get_yaxis().set_visible(False)
            coef = n_cols / n_rows
            delta_y = 0.05*coef
            ax.set_title(f'{variable_names[i]}--{variable_names[j]}', size=text_size)
            if masks is not None:
                current_mask = masks[ind]
            #delta_y = 0.06 * coef
            disable_yticks = j > (i+1)
            basic_plot_fc(
                ax,
                values[ind],
                #offset=offset,
                vmin=vmin,
                vmax=vmax,
                color_name=color_name,
                disable_yticks=disable_yticks,
                disable_margin_spaces=True
            )
            ind += 1
    
    fig.subplots_adjust(wspace=0., hspace=0.)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig

def si2_compact_plot_fc(
    si_path,
    network_module_names,
    n_non_fc_modules,
    values_name,
    variable_names,
    image_size=224,
    patch_size=16,
    n_channels=3,
    vmin=0,
    vmax=1,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    color_name='red',
    show=True,
    figsize=(5, 3)
):
    save_path = None
    colorbar_path = None
    #n_rows = len(row_names)
    n_modules = len(network_module_names)
    n_cols = len(variable_names)
    n_rows = n_cols
    for i_mn, module_name in enumerate(network_module_names):
        values, act_shape = sensitivity_analysis.visualize.get_values(
            si_path, module_name, values_name, values_func
        )
        values = np.reshape(values, (-1,) + act_shape)
        if i_mn < n_non_fc_modules:
            values = get_common_patch(values)
        print(module_name)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            colorbar_filename = f'{save_filename_base}_{module_name}_colorbar.pdf'
            colorbar_path = os.path.join(save_dirname, colorbar_filename)
        
        plot_si2_fc(
            values=values,
            n_rows=n_rows,
            n_cols=n_cols,
            variable_names=variable_names,
            figsize=figsize,
            #figsize=(0.65*n_cols, 0.65*n_rows),
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            color_name=color_name,
        )
        
def shpv_compact_plot(
    shpv_path,
    network_module_names,
    values_name,
    variable_names,
    image_size=224,
    patch_size=16,
    n_channels=3,
    vmin=0,
    vmax=1,
    normalize=True,
    eps=1e-20,
    figsize=None,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    cmap_name='Reds',
    show=True,
    show_colorbar=True,
    global_colorbounds=False
):
    save_path = None
    if global_colorbounds:
        #assert (vmin is None) and (vmax is None)
        if vmin is None:
            vmin = np.inf # vmin = vmin or np.inf, but sometimes better write it explicitly
        if vmax is None:
            vmax = -np.inf
        for i_mn, module_name in enumerate(network_module_names):
            values, act_shape = sensitivity_analysis.visualize.get_shp_values(
                shpv_path, module_name, values_name, normalize, values_func, eps
            )
            vmin = min(vmin, values.min())
            vmax = max(vmax, values.max())
    
    for i_mn, module_name in enumerate(network_module_names):
        values, act_shape = sensitivity_analysis.visualize.get_shp_values(
            shpv_path, module_name, values_name, normalize, values_func, eps
        )
        values = np.reshape(values, (-1, ) + act_shape)
        values = reconstruct_conv_projected_image(
            values, image_size, patch_size, n_channels
        )
        values = np.reshape(
            values, (1, -1, n_channels*image_size, image_size)
        )
        print(module_name)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
            colorbar_filename = f'{save_filename_base}_{module_name}_colorbar.pdf'
            colorbar_path = os.path.join(save_dirname, colorbar_filename)
 
        plot_decomposed_images(
            images=values,
            variable_names=variable_names,
            figsize=figsize,
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            block_names=("", ),
            cmap_name=cmap_name,
            show_colorbar=show_colorbar,
            colorbar_path=colorbar_path
        )

def shpv_compact_plot_fc(
    shpv_path,
    network_module_names,
    n_non_fc_modules,
    values_name,
    variable_names,
    image_size=224,
    patch_size=16,
    n_channels=3,
    vmin=0,
    vmax=1,
    normalize=True,
    eps=1e-20,
    figsize=None,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    color_name='red',
    show=True,
    global_colorbounds=False
):
    n_modules = len(network_module_names)
    n_cols = len(variable_names)
    save_path = None
    if global_colorbounds:
        #assert (vmin is None) and (vmax is None)
        if vmin is None:
            vmin = np.inf # vmin = vmin or np.inf, but sometimes better write it explicitly
        if vmax is None:
            vmax = -np.inf
        for i_mn, module_name in enumerate(network_module_names):
            values, act_shape = sensitivity_analysis.visualize.get_shp_values(
                shpv_path, module_name, values_name, normalize, values_func, eps
            )
            if i_mn < n_non_fc_modules:
                values = np.reshape(values, (-1,) + act_shape)
                values = get_common_patch(values)
            vmin = min(vmin, values.min())
            vmax = max(vmax, values.max())
    
    for i_mn, module_name in enumerate(network_module_names):
        values, act_shape = sensitivity_analysis.visualize.get_shp_values(
            shpv_path, module_name, values_name, normalize, values_func, eps
        )
        if i_mn < n_non_fc_modules:
            values = np.reshape(values, (-1,) + act_shape)
            values = get_common_patch(values)
        values = np.reshape(
            values, (n_cols, -1)
        )
        print(module_name)
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
        fig, ax = plt.subplots(1, n_cols, figsize=figsize)
        for i_var in range(n_cols):
            cax = ax[i_var]
            cax.set_title(f"{variable_names[i_var]}")
            disable_yticks = i_var > 0
            basic_plot_fc(
                cax,
                values[i_var],
                #offset=offset,
                vmin=vmin,
                vmax=vmax,
                color_name=color_name,
                disable_yticks=disable_yticks,
                disable_margin_spaces=True
            )
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.clf()
        
'''
    print(s)
    c, s = get_values(datapath, s, "si+sT")
    n_h = int((s[0] - 1)**0.5)
    for k in (0, 1):
        print(c.shape, s)
        #r = np.reshape(c[k], (-1, s[0], s[1]))[:, 1:, :]
        #r = np.reshape(r, (-1, n_h, n_h, s[1]))
        r = np.reshape(c[k], (-1, s[0], s[1]))
        print(r.shape)
        r = reconstruct_conv_projected_image(
            r[:, 1:, :], 224, 16
        )
        print(r.shape)
        #r = reconstruct_conv_projected_image(
        #    r.reshape((r.shape[0]*r.shape[1], s[0], s[1]), order='C')[:, 1:, :], 224, 16
        #)
        fig, ax = plt.subplots(1, r.shape[0], figsize=(10, 4))
        for i in range(r.shape[0]):
            w = r[i].reshape((r.shape[1]*r.shape[2], -1))
            #wmin = np.min(w, axis=(-2, -1), keepdims=True)
            #wmax = np.max(w, axis=(-2, -1), keepdims=True)
            #rn = (r - rmin) / (rmax - rmin)
            ax[i].imshow(np.clip(w, 0., 1.), cmap="Reds")
            ax[i].axis("off")
        plt.tight_layout()
        plt.show()
''';
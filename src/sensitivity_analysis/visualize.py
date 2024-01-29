import os

import numpy as np
import matplotlib as mpl


import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.cm

import h5py

import sensitivity_analysis.shapley

# todo:
# [ok - si, si2; not ok - shpv/tv] 1) grid_spec inside grid_spec
# [ok - si, si2; not ok - shpv/tv] 2) margins between activations of filters
# [okke] 3) hstograms of values
# [okke] 4) saving/showing flags
# [okke] 5) fc pl'ot: remove (0.0, 1.0) from xticks, display yticks once

#######################
# 1) basic plots




def plot_histograms(
    value_arrays,
    value_names,
    figsize=(10, 3),
    save_path=None,
    show=True,
    fc='#ada493',
    ec='#9e937f'
):
    # https://icolorpalette.com/ash-and-cod-gray
    nrows = 1
    ncols = len(value_arrays)    
    assert len(value_names) == ncols
    
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(ncols):
        ax[i].hist(
            value_arrays[i].flatten(), bins='auto', fc=fc, ec=ec,
            alpha=0.2, lw=1
        )
        ax[i].set_title(value_names[i])
        ax[i].set_xlabel('Estimated values (bins)')
        ax[i].set_ylabel('Counts')
        ax[i].grid(alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig

def basic_plot_fc(
    fig,
    gs,
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
    ax = fig.add_subplot(gs)
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
    ax.set_xlim(
        left=images.min() if vmin is None else vmin,
        right=images.max() if vmax is None else vmax
    )
    ax.invert_yaxis()
    ax.grid(alpha=0.5)
    
    #xticklabels = ax.get_xticklabels()
    #xticklabels[0] = xticklabels[-1] = ''
    #ax.set_xticklabels(xticklabels)
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
    
    return ax, im

def basic_plot_conv2d(
    fig,
    gs,
    images,
    n_rows,
    n_cols,
    masks=None,
    offset=1,
    vmin=None,
    vmax=None,
    cmap_name='Reds'
):
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_transparency_blend.html
    ind = 0
    axes, ims = [], []
    batch_size = len(images)
    
    _, _, height, width  = images.shape
    
    inner_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, gs, wspace=0, hspace=0
    )
    local_mask = None
    if masks is not None:
        bg_array = np.full((height, width, 3), 128, dtype=np.uint8)

    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(inner_gs[i, j])
            if ind < batch_size:
                if masks is not None:
                    mask_im = ax.imshow(bg_array)
                    local_mask = masks[ind][0]
                    #loc_bg_array = bg_array.copy()
                    #loc_bg_array[1-local_mask
                im = ax.imshow(
                    np.transpose(images[ind], (1, 2, 0)),
                    #cmap='coolwarm',
                    alpha=local_mask,
                    cmap=cmap_name,
                    interpolation='none',
                    vmin=images.min() if vmin is None else vmin,
                    vmax=images.max() if vmax is None else vmax,
                )                
                ims.append(im)
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            axes.append(ax)
            ind += offset
    return axes, ims

#######################
# 2) 


def plot_decomposed_images(
    images,
    variable_names,
    n_rows,
    n_cols,
    n_col_blocks,
    masks=None,
    layer_type='conv2d',
    offset=1,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    block_names=None,
    conv2d_cmap_name='Reds',
    mask_conv2d_cmap_name='Reds',
    fc_color_name='red',
    mask_color_name='firebrick',
    show_colorbar=True,
    colorbar_path=None,
):
    assert layer_type in ['conv2d', 'fc']
    nvars = len(variable_names)
    nblocks = len(images)
    assert nblocks == n_col_blocks
    if block_names is not None:
        assert len(block_names) == nblocks
    
    batch_size = images.shape[0]
    offset = batch_size // (n_rows*n_cols)
    offset = max(1, offset)
    if figsize is None:
        figsize = (1.5*nvars, 1.5*nblocks)
        
    if show_colorbar and (layer_type == 'conv2d'):
        assert (vmin is not None) and (vmax is not None)
        cb_fig = plot_colorbar(
            figsize=(figsize[0], 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax,
            dv=(vmax-vmin)/5, ticks=None, label='', cmap=conv2d_cmap_name
        )
        #plt.tight_layout()
        if colorbar_path is not None:
            plt.savefig(colorbar_path, bbox_inches='tight')
        if show:
            plt.show(cb_fig)
        else:
            plt.clf()
    
    if layer_type == 'conv2d':
        #height_ratios = np.ones(nsi+1)
        #height_ratios[0] = 0.1#height_ratios[-1] = 0.2
        #height_ratios[-1] = 0.05#height_ratios[-1] = 0.2
        #width_ratios = np.ones(nvars+1)
        #width_ratios[0] = 0.1#width_ratios[-1] = 0.2
        #width_ratios[-1] = 0.025#width_ratios[-1] = 0.2
        
        coef_y = 0.1*n_cols
        coef_x = 0.1*n_rows
    
    elif layer_type == 'fc':
        #assert n_rows == 1 == n_cols
        #n_rows = n_cols = 1
        #height_ratios = np.ones(n_rows+1)
        #height_ratios[0] = 0.1
        #height_ratios[-1] = 0.05
        #width_ratios = np.ones(n_cols+1)
        #width_ratios[0] = 0.1
        #width_ratios[-1] = 0.05
        #width_ratios[1] = 1.2
        
        coef_y = 0.0125*n_cols
        coef_x = 0.0125*n_rows
        
    else:
        raise ValueError
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    #nvars = int((1 + (1+8*images.shape[0])**0.5)/2)
    
    #top = 1 - i/nsi
    #bottom = 1 - (i+1)/nsi
    #left = j/nvars
    #right = (j+1)/nvars
    gs = fig.add_gridspec(
        #nrows=nsi+2, ncols=nvars+2,# left=left, right=right,
        #nrows=nsi+1, ncols=nvars+1,# left=left, right=right,
        nrows=nblocks, ncols=nvars,# left=left, right=right,
        #top=top, bottom=bottom,
        wspace=0.05, hspace=0.05,
        #height_ratios=height_ratios, width_ratios=width_ratios
    )
    #figure_aspect_w2h = (top-bottom)/(right-left)
    #figure_aspect_w2h = (top-bottom)/(right-left)
    #figure_width = width / height * figure_height * n_cols / (n_rows*figure_aspect_w2h)
    #figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp
    
    current_mask = None
    text_size = 'large'
    
    ind = 0
    for i in range(nblocks):
        #ax = plt.subplot(gs[i+1, 0], frameon=False)
        ax = plt.subplot(gs[i, 0], frameon=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        #ax.text(0.8, 0.5, row_names[i], size=text_size)
        if block_names is not None:
            ax.text(-0.15, 0.5, block_names[i], size=text_size)
        
        for j in range(nvars):
            
            #local_gs = gs[i+1, j+1]
            local_gs = gs[i, j]
            if i == 0:
                #ax = plt.subplot(gs[i, j+1], frameon=False)
                ax = plt.subplot(gs[i, j], frameon=False)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                #ax.text(0.05, 0.9, f'{variable_names[j]}', size=text_size)
                ax.text(0.05, 1+coef_y/n_rows, f'{variable_names[j]}', size=text_size)
            
            if masks is not None:
                current_mask = masks[i, j]
            
            if layer_type == 'conv2d':
                axes, ims = basic_plot_conv2d(
                    #fig, gs[i+1, j+1], images[i, j], n_rows, n_cols, offset, vmin, vmax
                    fig,
                    local_gs,
                    images[i, j],
                    n_rows, 
                    n_cols,
                    masks,
                    offset,
                    vmin,
                    vmax,
                    cmap_name=conv2d_cmap_name
                )
            elif layer_type == 'fc':
                disable_yticks = j > 0
                ax, im = basic_plot_fc(
                    fig,
                    local_gs,
                    images[i, j],
                    masks=None or current_mask,
                    offset=offset,
                    vmin=vmin,
                    vmax=vmax,
                    color_name=fc_color_name,
                    mask_color_name=mask_color_name,
                    disable_yticks=disable_yticks,
                    disable_margin_spaces=True
                )
            else:
                raise ValueError

    #plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
    return fig
            
    #plt.setp(axes, title=f'{variable_names[i]}--{variable_names[j]}')
    # 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    

    
####################################################
####################################################
    
def plot_si2_images(
    images,
    variable_names,
    n_rows,
    n_cols,
    masks=None,
    layer_type='conv2d',
    offset=1,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    conv2d_cmap_name='Reds',
    mask_conv2d_cmap_name='Reds',
    fc_color_name='red',
    mask_color_name='firebrick',
    show_colorbar=True,
    rescale_blocks=False
):
    assert layer_type in ['conv2d', 'fc']
    #(36, )
    batch_size = images.shape[0]
    offset = batch_size // (n_rows*n_cols)
    offset = max(1, offset)
    if figsize is None:
        figsize = (1.5*n_cols, 1.5*n_rows)
    nvars = len(variable_names)
    
    if show_colorbar and (layer_type == 'conv2d'):
        assert (vmin is not None) and (vmax is not None) and (show)
        cb_fig = plot_colorbar(
            figsize=(figsize[0], 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax,
            dv=(vmax-vmin)/5, ticks=None, label='', cmap=conv2d_cmap_name
        )
        plt.show(cb_fig)
    
    if layer_type == 'conv2d':
        height_ratios = np.ones(n_rows+2)
        #height_ratios[0] = 0.25#height_ratios[-1] = 0.2
        #height_ratios[-1] = 0.25#height_ratios[-1] = 0.2
        width_ratios = np.ones(n_cols+2)
        #width_ratios[0] = 0.25#width_ratios[-1] = 0.2
        #width_ratios[-1] = 0.25#width_ratios[-1] = 0.2
        coef_y = 0.05*n_cols
        coef_x = 0.05*n_rows
    
    elif layer_type == 'fc':
        #assert n_rows == 1 == n_cols
        n_rows = n_cols = 1
        height_ratios = np.ones(n_rows+2)
        #height_ratios[0] = 0.05
        #height_ratios[-1] = 0.05
        width_ratios = np.ones(n_cols+2)
        #width_ratios[0] = 0.05
        #width_ratios[-1] = 0.05
        #width_ratios[1] = 1.2
        coef_y = 0.0125*n_cols
        coef_x = 0.0125*n_rows
    else:
        raise ValueError
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        #nrows=nsi+2, ncols=nvars+2,# left=left, right=right,
        #nrows=nsi+1, ncols=nvars+1,# left=left, right=right,
        nrows=nvars, ncols=nvars,# left=left, right=right,
        #top=top, bottom=bottom,
        wspace=0.1, hspace=0.1,
        #height_ratios=height_ratios, width_ratios=width_ratios
    )
    
    text_size = 'large'
    current_mask = None
    
    n_col_blocks = n_row_blocks = nvars
    
    #nvars = int((1 + (1+8*images.shape[0])**0.5)/2)
    ind = 0
    for i in range(nvars):
        for j in range(nvars):
            
            local_gs = gs[i, j]
            local_max_width = figsize[0] / n_col_blocks
            local_max_height = figsize[1] / n_row_blocks
            #if rescale_blocks:
            h_a = local_max_height / n_rows
            w_a = local_max_width / n_cols
            a = min(h_a, w_a)
            local_height = a*n_rows / local_max_height
            #else:
            #    local_height = n_rows / max_n_rows
            #    a = local_max_height / max_n_rows
            #    #local_width = a*np.floor(local_max_width / a) / local_max_width
            local_width = a*n_cols / local_max_width
            
            sgc = local_gs.subgridspec(
                nrows=2,
                ncols=2,
                height_ratios=(local_height, 1-local_height),
                width_ratios=(local_width, 1-local_width)
                #wspace=0.25,
                #hspace=0,
            )
            local_subsgc = sgc[0, 0]
            
            #gs = fig.add_gridspec(
            #    nrows=n_rows+2, ncols=n_cols+2, left=i/nvars, right=(i+1)/nvars,
            #    top=1 - j/nvars, bottom=1 - (j+1)/nvars, wspace=0., hspace=0.,
            #    height_ratios=height_ratios, width_ratios=width_ratios
            #)
            #if i == j+1:
                #ax = plt.subplot(gs[i, j+1], frameon=False)
            if i < j:
                ax = plt.subplot(local_subsgc, frameon=False)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                #ax.text(0.05, 0.9, f'{variable_names[j]}', size=text_size)
                #ax.text(
                #    0.05,
                #    1+coef_y/n_rows,
                #    f'{variable_names[i]}--{variable_names[j]}',
                #    size=text_size
                #)
                coef = n_cols / n_rows
                delta_y = 0.05*coef
                ax.text(
                    0.05,
                    1+delta_y,
                    f'{variable_names[i]}--{variable_names[j]}',
                    size=text_size
                )
                
                if masks is not None:
                    current_mask = masks[ind]
                
                if layer_type == 'conv2d':
                    axes, ims = basic_plot_conv2d(
                        fig,
                        local_gs,
                        images[ind],
                        n_rows,
                        n_cols,
                        masks,
                        offset,
                        vmin,
                        vmax,
                        cmap_name=conv2d_cmap_name
                    )
                elif layer_type == 'fc':
                    disable_yticks = j > i+1
                    ax, im = basic_plot_fc(
                        fig,
                        local_gs,
                        images[ind],
                        masks=None or current_mask,
                        offset=offset,
                        vmin=vmin,
                        vmax=vmax,
                        color_name=fc_color_name,
                        mask_color_name=mask_color_name,
                        disable_yticks=disable_yticks,
                        disable_margin_spaces=True
                    )
                else:
                    raise ValueError
                ind += 1
                
    
    #fig.subplots_adjust(right=0.8)
    #cax = fig.add_axes([1.05, 0.05, 0.1, 0.9])
    #fig.subplots_adjust(right=0.8)
    #fig.colorbar(ims[-1], cax=cax)
    
    fig.subplots_adjust(wspace=0., hspace=0.)
    #plt.tight_layout()
    #plt.show()
    #plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig

#############
'''
def plot_tv_images(
    images,
    n_rows,
    n_cols,
    masks=None,
    layer_type='conv2d',
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    block_names=None,
    conv2d_cmap_name='Reds',
    mask_conv2d_cmap_name='Reds',
    fc_color_name='red',
    mask_color_name='firebrick'
):
    assert layer_type in ['conv2d', 'fc']
    
    n_images = images[0].shape[0]
    offset = n_images // (n_rows*n_cols)
    offset = max(1, offset)
    
    if figsize is None:
        figsize = (1.5, 1.5)
    
    if vmin is None:
        vmin = images.min()
    if vmax is None:
        vmax = images.max()
    
    n_blocks = len(images)
    if block_names is not None:
        assert len(block_names) == n_blocks
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    #nvars = int((1 + (1+8*images.shape[0])**0.5)/2)
    
    #top = 1 - i/nsi
    #bottom = 1 - (i+1)/nsi
    #left = j/nvars
    #right = (j+1)/nvars
    gs = fig.add_gridspec(
        #nrows=nsi+2, ncols=nvars+2,# left=left, right=right,
        #nrows=nsi+1, ncols=nvars+1,# left=left, right=right,
        nrows=1, ncols=n_blocks,# left=left, right=right,
        #top=top, bottom=bottom,
        wspace=0.05, hspace=0.05,
        #height_ratios=height_ratios, width_ratios=width_ratios
    )
    
    current_mask = None
    text_size = 'large'
    coef = n_cols / n_rows
    
    for i_block in range(n_blocks):
        local_gs = gs[0, i_block]
        ax = plt.subplot(local_gs, frameon=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        if block_names is not None:
            #ax.text(0.05, 0.9, f'{variable_names[j]}', size=text_size)
            delta_y = 0.05*coef
            ax.text(0.05, 1+delta_y, block_names[i_block], size=text_size)
            
        if masks is not None:
            current_mask = masks[i_block]
            
        if layer_type == 'conv2d':
            pcm = matplotlib.cm.ScalarMappable(
                matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                cmap=conv2d_cmap_name
            )
            delta_y = 0.06 * coef
            cax = ax.inset_axes([0., 1+delta_y, 1., 0.05*coef], transform=ax.transAxes)
            cb = fig.colorbar(pcm, ax=ax, cax=cax, orientation='horizontal')
            cb.ax.xaxis.set_ticks_position('top')
            axes, ims = basic_plot_conv2d(
                #fig, gs[i+1, j+1], images[i, j], n_rows, n_cols, offset, vmin, vmax
                fig,
                local_gs,
                images[i_block],
                n_rows,
                n_cols,
                masks,
                offset,
                vmin,
                vmax,
                cmap_name=conv2d_cmap_name
            )
        elif layer_type == 'fc':
            ax, im = basic_plot_fc(
                fig,
                local_gs,
                images[i_block],
                masks=None or current_mask,
                offset=offset,
                vmin=vmin,
                vmax=vmax,
                color_name=fc_color_name,
                mask_color_name=mask_color_name,
                disable_yticks=False,
                disable_margin_spaces=True
            )
            #fig, gs[i+1, j+1], images[i, j], offset, vmin, vmax
        else:
            raise ValueError

    #plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig
'''

def plot_many_block_images_scaled(
    images,
    block_names,
    n_rows, # number of single activation maps
    n_cols, # number of single activation maps
    n_col_blocks, # activation maps within 1 variable
    masks=None,
    layer_type='conv2d',
    offset=1,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    conv2d_cmap_name='Reds',
    mask_conv2d_cmap_name='Reds',
    fc_color_name='red',
    mask_color_name='firebrick',
    rescale_blocks=True,
    show_colorbar=True,
    colorbar_path=None,
):
    
    assert layer_type in ['conv2d', 'fc']
    # (36, )
    batch_size = images.shape[0]
    offset = batch_size // (n_rows*n_cols)
    offset = max(1, offset)
    if figsize is None:
        figsize = 1.5
    if isinstance(figsize, int):
        figsize = (figsize*n_cols, figsize*n_rows)
    Nblocks = len(block_names)
    if n_col_blocks is None:
        n_col_blocks = Nblocks # no so good: nblocks = variables, n_col_blocks = n_columns of activation parts
    n_row_blocks = int(np.ceil(Nblocks / n_col_blocks))
    
    #if layer_type == 'conv2d':
    #    height_ratios = np.ones(n_rows+2)
    #    width_ratios = np.ones(n_cols+2)
    #    coef_y = 0.05*n_cols
    #    coef_x = 0.05*n_rows
    #
    #elif layer_type == 'fc':
    #    n_rows = n_cols = 1
    #    height_ratios = np.ones(n_rows+2)
    #    width_ratios = np.ones(n_cols+2)
    #    coef_y = 0.0125*n_cols
    #    coef_x = 0.0125*n_rows
    #else:
    #    raise ValueError
    if show_colorbar and (layer_type == 'conv2d'):
        assert (vmin is not None) and (vmax is not None)
        cb_fig = plot_colorbar(
            figsize=(figsize[0], 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax,
            dv=(vmax-vmin)/5, ticks=None, label='', cmap=conv2d_cmap_name
        )
        if colorbar_path is not None:
            plt.savefig(colorbar_path, bbox_inches='tight')
        if show:
            plt.show(cb_fig)
        else:
            plt.clf()
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        nrows=n_row_blocks,
        ncols=n_col_blocks,
        wspace=0.05,
        hspace=0.05,
        #height_ratios=(1-fc_height_fraction, fc_height_fraction)
    )
    text_size = 'large'
    current_mask = None
    
    ind = 0
    for i_block_row in range(n_row_blocks):
        for i_block_col in range(n_col_blocks):
            #if i_block_row*Nblocks_per_row + i_block_col >= Nblocks:
            if ind >= Nblocks:
                break
            local_gs = gs[i_block_row, i_block_col]
            #sgc = local_gs.subgridspec(
            #    nrows=2,
            #    ncols=1, # n_col_blocks >= n_row_blocks
            #    wspace=0.25,
            #    hspace=0.25,
            #)
            local_max_width = figsize[0] / n_col_blocks
            local_max_height = figsize[1] / n_row_blocks
            if rescale_blocks:
                h_a = local_max_height / n_rows
                w_a = local_max_width / n_cols
                a = min(h_a, w_a)
                local_height = a*n_rows / local_max_height
            else:
                local_height = n_rows_list[i_module] / max_n_rows
                a = local_max_height / max_n_rows
                #local_width = a*np.floor(local_max_width / a) / local_max_width
            local_width = a*n_cols / local_max_width
            
            sgc = local_gs.subgridspec(
                nrows=2,
                ncols=2,
                height_ratios=(local_height, 1-local_height),
                width_ratios=(local_width, 1-local_width)
                #wspace=0.25,
                #hspace=0,
            )
            local_subsgc = sgc[0, 0]
            
            ax = plt.subplot(local_subsgc, frameon=False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
            coef = n_cols / n_rows
            delta_y = 0.05*coef
            ax.text(0.05, 1+delta_y, block_names[ind], size=text_size)
            
            if layer_type == 'conv2d':
                pcm = matplotlib.cm.ScalarMappable(
                    matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                    cmap=conv2d_cmap_name
                )
                delta_y = 0.06 * coef
                #cax = ax.inset_axes(
                #    [0., 1+delta_y, 1., 0.05*coef], transform=ax.transAxes
                #)
                #cb = fig.colorbar(pcm, ax=ax, cax=cax, orientation='horizontal')
                #cb.ax.xaxis.set_ticks_position('top')
                axes, ims = basic_plot_conv2d(
                    fig,
                    local_subsgc,
                    images[ind],
                    n_rows,
                    n_cols,
                    offset=offset,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_name=conv2d_cmap_name
                )
            elif layer_type == 'fc':
                disable_yticks = i_block_col > 0
                ax, im = basic_plot_fc(
                    fig,
                    local_subsgc,
                    images[ind],
                    offset=offset,
                    vmin=vmin,
                    vmax=vmax,
                    color_name=fc_color_name,
                    disable_yticks=disable_yticks,
                    disable_margin_spaces=True
                )
            else:
                raise ValueError
            ind += 1
    #plt.tight_layout()
    fig.subplots_adjust(wspace=0., hspace=0.)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
    return fig
            

def plot_many_block_images(
    images,
    block_names,
    n_rows,
    n_cols,
    n_col_blocks=None,
    masks=None,
    layer_type='conv2d',
    offset=1,
    figsize=None,
    vmin=None,
    vmax=None,
    save_path=None,
    show=True,
    conv2d_cmap_name='Reds',
    mask_conv2d_cmap_name='Reds',
    fc_color_name='red',
    mask_color_name='firebrick',
    show_colorbar=True
):
    
    assert layer_type in ['conv2d', 'fc']
    # (36, )
    batch_size = images.shape[0]
    offset = batch_size // (n_rows*n_cols)
    offset = max(1, offset)
    if figsize is None:
        figsize = (1.5*n_cols, 1.5*n_rows)
    nblocks = len(block_names)
    if n_col_blocks is None:
        n_col_blocks = nblocks
    n_row_blocks = int(np.ceil(nblocks / n_col_blocks))
    
    if layer_type == 'conv2d':
        height_ratios = np.ones(n_rows+2)
        #height_ratios[0] = 0.25#height_ratios[-1] = 0.2
        #height_ratios[-1] = 0.25#height_ratios[-1] = 0.2
        width_ratios = np.ones(n_cols+2)
        #width_ratios[0] = 0.25#width_ratios[-1] = 0.2
        #width_ratios[-1] = 0.25#width_ratios[-1] = 0.2
        coef_y = 0.05*n_cols
        coef_x = 0.05*n_rows
    
    elif layer_type == 'fc':
        n_rows = n_cols = 1
        height_ratios = np.ones(n_rows+2)
        #height_ratios[0] = 0.05
        #height_ratios[-1] = 0.05
        width_ratios = np.ones(n_cols+2)
        #width_ratios[0] = 0.05
        #width_ratios[-1] = 0.05
        #width_ratios[1] = 1.2
        coef_y = 0.0125*n_cols
        coef_x = 0.0125*n_rows
    else:
        raise ValueError
        
    if show_colorbar and (layer_type == 'conv2d') and show:
        assert (vmin is not None) and (vmax is not None) and (show)
        cb_fig = plot_colorbar(
            figsize=(30, 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax,
            dv=(vmax-vmin)/5, ticks=None, label='', cmap=conv2d_cmap_name
        )
        plt.show(cb_fig)
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = fig.add_gridspec(
        #nrows=nsi+2, ncols=nvars+2,# left=left, right=right,
        #nrows=nsi+1, ncols=nvars+1,# left=left, right=right,
        nrows=n_row_blocks, ncols=n_col_blocks,# left=left, right=right,
        #top=top, bottom=bottom,
        wspace=0.1, hspace=0.1,
        #height_ratios=height_ratios, width_ratios=width_ratios
    )
    
    text_size = 'large'
    current_mask = None
    
    #nvars = int((1 + (1+8*images.shape[0])**0.5)/2)
    ind = 0
    for i in range(n_row_blocks):
        for j in range(n_col_blocks):
            local_gs = gs[i, j]
            #gs = fig.add_gridspec(
            #    nrows=n_rows+2, ncols=n_cols+2, left=i/nvars, right=(i+1)/nvars,
            #    top=1 - j/nvars, bottom=1 - (j+1)/nvars, wspace=0., hspace=0.,
            #    height_ratios=height_ratios, width_ratios=width_ratios
            #)
            #if i == j+1:
                #ax = plt.subplot(gs[i, j+1], frameon=False)
            ax = plt.subplot(local_gs, frameon=False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if ind < nblocks:
                ax.text(
                    0.05,
                    1+coef_y/n_rows,
                    f'{block_names[ind]}',
                    size=text_size
                )
                
                if masks is not None:
                    current_mask = masks[ind]
                
                
                if layer_type == 'conv2d':
                    axes, ims = basic_plot_conv2d(
                        fig,
                        local_gs,
                        images[ind],
                        n_rows,
                        n_cols,
                        current_mask,
                        offset,
                        vmin,
                        vmax,
                        cmap_name=conv2d_cmap_name
                    )
                elif layer_type == 'fc':
                    disable_yticks = j > 0
                    ax, im = basic_plot_fc(
                        fig,
                        local_gs,
                        images[ind],
                        masks=None or current_mask,
                        offset=offset,
                        vmin=vmin,
                        vmax=vmax,
                        color_name=fc_color_name,
                        mask_color_name=mask_color_name,
                        disable_yticks=disable_yticks,
                        disable_margin_spaces=True
                    )
                else:
                    raise ValueError
                ind += 1
                
    
    #fig.subplots_adjust(right=0.8)
    #cax = fig.add_axes([1.05, 0.05, 0.1, 0.9])
    #fig.subplots_adjust(right=0.8)
    #fig.colorbar(ims[-1], cax=cax)
    
    fig.subplots_adjust(wspace=0., hspace=0.)
    #plt.tight_layout()
    #plt.show()
    #plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig
    
    
####################################
####################################

'''
def cs_plot_blocks(
    cs_paths_dict,
    network_module_names,
    augmentations_list,
    values_name,
    n_conv_modules,
    n_rows,
    n_cols,
    n_col_blocks,
    vmin=0,
    vmax=1,
    func_values=None,
    save_filename_base=None,
    save_dirname=None,
    conv2d_cmap_name='Reds',
    fc_color_name='red'
):
    save_path = None
    n_augs = len(augmentations_list)
    for i_mn, module_name in enumerate(network_module_names):
        values = []
        for i_aug, aug_name in enumerate(augmentations_list):
            with h5py.File(cs_paths_dict[aug_name], 'r') as cs:
                cs_values = cs[f'{module_name}/{values_name}'][:]
                act_shape = tuple(cs.attrs[f'{module_name}'])
            values.append(cs_values)
        values = np.array(values)
        if func_values is not None:
            values = func_values(values)
        print(module_name)
        n_rows = act_shape[0] // n_cols
        if i_mn < n_conv_modules:
            values = values.reshape((n_augs, ) + act_shape)
            values = values[:, :, None, :, :]
            layer_type = 'conv2d'
        else:
            values = values.reshape((n_augs, -1, act_shape[0]))
            layer_type = 'fc'
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
        #sensitivity_analysis.visualize.plot_many_block_images(
        plot_many_block_images(
            values,
            block_names=augmentations_list,
            n_rows=n_rows,
            n_cols=n_cols,
            n_col_blocks=n_col_blocks,
            layer_type=layer_type,
            offset=1,
            figsize=((n_cols*n_col_blocks)//2, n_rows),
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=True,
            conv2d_cmap_name=conv2d_cmap_name,
            fc_color_name=fc_color_name
        )
'''

def cs_compact_plot_blocks(
    cs_paths_dict,
    network_module_names,
    augmentations_list,
    values_name,
    n_conv_modules,
    n_cols,
    n_col_blocks,
    n_rows_fc=None,
    vmin=0,
    vmax=1,
    values_func=None,
    post_values_func=None,
    masks_paths_dict=None,
    masking_func=None,
    post_masking_func=None,
    masking_values_name=None,
    save_filename_base=None,
    save_dirname=None, 
    conv2d_cmap_name='Reds',
    fc_color_name='red',
    mask_color_name='firebrick',
    show=True,
    show_colorbar=True,
    global_colorbounds=False,
    global_bounds_conv_only=True
):
    assert (masks_paths_dict is None) == (masking_func is None) == (masking_values_name is None)
    if global_colorbounds:
        #assert (vmin is None) and (vmax is None)
        if vmin is None:
            vmin = np.inf # vmin = vmin or np.inf, but sometimes better write it explicitly
        if vmax is None:
            vmax = -np.inf
        for i_mn, module_name in enumerate(network_module_names):
            if global_bounds_conv_only and (i_mn == n_conv_modules):
                break
            for i_aug, aug_name in enumerate(augmentations_list):
                values, act_shape = get_values(
                    cs_paths_dict[aug_name], module_name, values_name, values_func
                )
                vmin = min(vmin, values.min())
                vmax = max(vmax, values.max())
            
            
    save_path = None
    n_augs = len(augmentations_list)
    masks = None
    for i_mn, module_name in enumerate(network_module_names):
        values = []
        if masks_paths_dict is not None:
            masks = []
        for i_aug, aug_name in enumerate(augmentations_list):
            cs_values, act_shape = get_values(
                cs_paths_dict[aug_name], module_name, values_name, values_func
            )
            values.append(cs_values)
            if masks_paths_dict is not None:
                ms_values, ms_shape = get_values(
                    masks_paths_dict[aug_name], module_name, masking_values_name, masking_func
                )
                act_shape == tuple(ms_shape)
                masks.append(ms_values)
        values = np.array(values)
        if post_values_func is not None:
            values = post_values_func(values)
        if masks_paths_dict is not None:
            masks = np.array(masks)
            if post_masking_func is not None:
                masks = post_masking_func(masks)
        print(module_name)
        if i_mn < n_conv_modules:
            n_rows = act_shape[0] // n_cols
            values = values.reshape((n_augs, ) + act_shape)
            values = values[:, :, None, :, :]
            layer_type = 'conv2d'
            if masks_paths_dict is not None:
                masks = masks.reshape((n_augs, ) + act_shape)
                masks = masks[:, :, None, :, :]
        else:
            n_rows = n_rows_fc or max(1, act_shape[0] // (4*n_cols))
            values = values.reshape((n_augs, -1, act_shape[0]))
            layer_type = 'fc'
            if masks_paths_dict is not None:
                masks = masks.reshape((n_augs, -1, act_shape[0]))
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
        
        '''
        if i_mn < n_conv_modules and show_colorbar:
            assert (vmin is not None) and (vmax is not None) and show
            cb_fig = plot_colorbar(
                figsize=(30, 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax, dv=(vmax-vmin)/5, ticks=None,
                label='', cmap=conv2d_cmap_name
            )
            plt.show(cb_fig)
        '''
        
        #sensitivity_analysis.visualize.plot_many_block_images(
        plot_many_block_images(            
            values,
            block_names=augmentations_list,
            n_rows=n_rows,
            n_cols=n_cols,
            n_col_blocks=n_col_blocks,
            masks=masks,
            layer_type=layer_type,
            offset=1,
            figsize=((n_cols*n_col_blocks)//2, n_rows),
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            conv2d_cmap_name=conv2d_cmap_name,
            fc_color_name=fc_color_name,
            mask_color_name=mask_color_name,
            show_colorbar=show_colorbar
        )

    
def custom_plot_single_value_images(
    values_path,
    network_module_names,
    values_name,
    n_conv_modules,
    n_cols,
    n_col_blocks,
    n_rows_fc=None,
    vmin=None, #########################
    vmax=None, #########################
    values_func=None,
    post_values_func=None,
    save_path=None,
    conv2d_cmap_name='Reds',
    fc_color_name='red',
    figsize=None,
    rescale_conv_blocks=True,
    fc_height_fraction=0.2,
    show_colorbar=True,
    global_colorbounds=False,
    global_bounds_conv_only=True,
    colorbar_path=None,
    show=True,
):
    if global_colorbounds:
        #assert (vmin is None) and (vmax is None)
        if vmin is None:
            vmin = np.inf # vmin = vmin or np.inf, but sometimes better write it explicitly
        if vmax is None:
            vmax = -np.inf
        for i_mn, module_name in enumerate(network_module_names):
            if global_bounds_conv_only and (i_mn == n_conv_modules):
                break
            values, act_shape = get_values(values_path, module_name, values_name, values_func)
            vmin = min(vmin, values.min())
            vmax = max(vmax, values.max())
    
    offset = 1
    n_rows_list = []
    values = []
    for i_mn, module_name in enumerate(network_module_names):
        current_values, act_shape = get_values(values_path, module_name, values_name, values_func)
        #with h5py.File(values_path, 'r') as vals:
        #    current_group = vals[module_name]
        #    if isinstance(values_name, str):
        #        current_values = current_group[values_name][:]
        #    else:
        #        current_values = []
        #        for vname in values_name:
        #            current_values.append(current_group[vname][:])
        #        current_values = np.array(current_values)
        #    act_shape = tuple(current_group.attrs['shape'])
        #    #act_shape = tuple(vals.attrs[f'{module_name}'])
        if i_mn < n_conv_modules:
            n_rows = act_shape[0] // n_cols
            current_values = current_values.reshape(act_shape)
            current_values = current_values[:, None, :, :]
            #if isinstance(values_name, str):
            #    current_values = current_values.reshape(act_shape)
            #    current_values = current_values[:, None, :, :]
            #else:
            #    current_values = current_values.reshape((-1, )+act_shape)
            #    current_values = current_values[:, :, None, :, :]
            layer_type = 'conv2d'
        else:
            n_rows = n_rows_fc or max(1, act_shape[0] // (4*n_cols))
            current_values = current_values.reshape((-1, act_shape[0]))
            #if isinstance(values_name, str):
            #    current_values = current_values.reshape((-1, act_shape[0]))
            #else:
            #    current_values = current_values.reshape((len(values_name), -1, act_shape[0]))
            layer_type = 'fc'
        n_rows_list.append( n_rows )
        values.append(current_values)    
    max_n_rows = max(n_rows_list[:n_conv_modules])
    if post_values_func is not None:
        values = post_values_func(values)
    
    if show_colorbar:
        assert (vmin is not None) and (vmax is not None)
        cb_fig = plot_colorbar(
            figsize=(30, 1), w=0.1, h=0.05, vmin=vmin, vmax=vmax, dv=(vmax-vmin)/5, ticks=None,
            label='', cmap=conv2d_cmap_name
        )
        if colorbar_path is not None:
            plt.savefig(colorbar_path, bbox_inches='tight')
        if show:
            plt.show(cb_fig)
        else:
            plt.clf()
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    n_block_rows = 2
    n_blocks_dict = {
        0: n_conv_modules,
        1: len(network_module_names) - n_conv_modules
    }
    
    n_block_cols = max(n_blocks_dict[0], n_blocks_dict[1])
    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        wspace=0.05,
        hspace=0.05,
        height_ratios=(1-fc_height_fraction, fc_height_fraction)
    )
    text_size = 'large'
    
    i_module = 0
    for i_block_row in range(n_block_rows):
        local_gs = gs[i_block_row, 0]
        #ax = plt.subplot(local_gs, frameon=False)
        sgc = local_gs.subgridspec(
            nrows=1,
            ncols=n_blocks_dict[i_block_row],
            wspace=0.25,
            hspace=0.25,
        )
        for i_block_col in range(n_block_cols):
            if n_blocks_dict[i_block_row] == i_block_col:
                break
        
            local_sgc = sgc[0, i_block_col]
            
            if i_module < n_conv_modules:
                local_max_width = figsize[0] / n_conv_modules
                local_max_height = figsize[1]*(1-fc_height_fraction) #/ 2
                if rescale_conv_blocks:
                    h_a = local_max_height / n_rows_list[i_module]
                    w_a = local_max_width / n_cols
                    a = min(h_a, w_a)
                    local_height = a*n_rows_list[i_module] / local_max_height
                    #if h_a < w_a:
                    #    local_height = 1
                    #    local_width = h_a*np.floor(local_max_width / h_a) / local_max_width
                    #else:
                    #    local_height = w_a*np.floor(local_max_height / w_a) / local_max_height
                    #    local_width = 1
                else:
                    local_height = n_rows_list[i_module] / max_n_rows
                    a = local_max_height / max_n_rows
                    #local_width = a*np.floor(local_max_width / a) / local_max_width
                local_width = a*n_cols / local_max_width
            else:
                local_height = 1
                local_width = 1
            subsgc = local_sgc.subgridspec(
                nrows=2,
                ncols=2,
                height_ratios=(local_height, 1-local_height),
                width_ratios=(local_width, 1-local_width)
                #wspace=0.25,
                #hspace=0,
            )
            local_subsgc = subsgc[0, 0]
            
            
            ax = plt.subplot(local_subsgc, frameon=False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            
            coef = n_cols / n_rows_list[i_module]
            delta_y = 0.05*coef
            ax.text(0.05, 1+delta_y, network_module_names[i_module], size=text_size)
            
            #if layer_type == 'conv2d':
            if i_block_row == 0:
                pcm = matplotlib.cm.ScalarMappable(
                    matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False),
                    cmap=conv2d_cmap_name
                )
                delta_y = 0.06 * coef
                #cax = ax.inset_axes(
                #    [0., 1+delta_y, 1., 0.05*coef], transform=ax.transAxes
                #)
                #cb = fig.colorbar(pcm, ax=ax, cax=cax, orientation='horizontal')
                #cb.ax.xaxis.set_ticks_position('top')
                axes, ims = basic_plot_conv2d(
                    fig,
                    local_subsgc,
                    values[i_module],
                    n_rows_list[i_module],
                    n_cols,
                    offset=offset,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_name=conv2d_cmap_name
                )
            #elif layer_type == 'fc':
            elif i_block_row == 1:
                vmax = None
                ax, im = basic_plot_fc(
                    fig,
                    local_subsgc,
                    values[i_module],
                    offset=offset,
                    vmin=vmin,
                    vmax=vmax,
                    color_name=fc_color_name,
                    disable_yticks=False,
                    disable_margin_spaces=True
                )
            else:
                raise ValueError
            i_module += 1
    #plt.tight_layout()
    #fig.subplots_adjust(wspace=0., hspace=0.)
    #plt.tight_layout()
    #plt.show()
    #plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.clf()
    return fig
    # figsize=((n_cols*n_col_blocks)//2, n_rows),
    
def si_compact_plot(
    si_path,
    network_module_names,
    values_name,
    variable_names,
    row_names,
    n_conv_modules,
    n_cols,
    n_row_blocks,
    n_rows_fc=None,
    vmin=0,
    vmax=1,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    conv2d_cmap_name='Reds',
    fc_color_name='red',
    show=True,
    show_colorbar=True
):
    save_path = None
    for i_mn, module_name in enumerate(network_module_names):
        with h5py.File(si_path, 'r') as si:
            values = si[f'{module_name}/{values_name}'][:]
            #act_shape = tuple(si.attrs[f'{module_name}'])
            act_shape = tuple(si[f'{module_name}'].attrs['shape'])
        if values_func is not None:
            values = values_func(values)
        print(module_name)
        if i_mn < n_conv_modules:
            n_rows = act_shape[0] // n_cols
            values = values.reshape((n_row_blocks, -1, ) + act_shape)
            values = values[:, :, :, None, :, :]
            layer_type = 'conv2d'
        else:
            n_rows = n_rows_fc or max(1, act_shape[0] // (4*n_cols))
            values = values.reshape((n_row_blocks, -1, act_shape[0]))
            layer_type = 'fc'
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
        
        plot_decomposed_images(
            images=values,
            variable_names=variable_names,
            n_rows=n_rows,
            n_cols=n_cols,
            n_col_blocks=len(values),
            layer_type=layer_type,
            offset=1,
            figsize=(n_cols//2*len(variable_names), n_rows//2*2),
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            block_names=row_names,
            conv2d_cmap_name=conv2d_cmap_name,
            fc_color_name=fc_color_name,
            show_colorbar=show_colorbar,
            colorbar_path=os.path.join(save_dirname, f'{save_filename_base}_{module_name}_colorbar.pdf'),
        )
        
        
def si2_compact_plot(
    si_path,
    network_module_names,
    values_name,
    variable_names,
    n_conv_modules,
    n_cols,
    #n_row_blocks,
    n_rows_fc=None,
    vmin=0,
    vmax=1,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    conv2d_cmap_name='Reds',
    fc_color_name='red',
    show=True,
    show_colorbar=True,
    rescale_blocks=False
):
    save_path = None
    for i_mn, module_name in enumerate(network_module_names):
        with h5py.File(si_path, 'r') as si:
            values = si[f'{module_name}/{values_name}'][:]
            #act_shape = tuple(si.attrs[f'{module_name}'])
            act_shape = tuple(si[f'{module_name}'].attrs['shape'])
        if values_func is not None:
            values = values_func(values)
        print(module_name)
        if i_mn < n_conv_modules:
            n_rows = act_shape[0] // n_cols
            values = values.reshape((-1, ) + act_shape)
            values = values[:, :, None, :, :]
            layer_type = 'conv2d'
        else:
            n_rows = n_rows_fc or max(1, act_shape[0] // (4*n_cols))
            values = values.reshape((-1, act_shape[0]))
            layer_type = 'fc'
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
        
        nvars = len(variable_names)
        plot_si2_images(
            images=values,
            variable_names=variable_names,
            n_rows=n_rows if layer_type == 'conv2d' else n_rows_fc,
            n_cols=n_cols if layer_type == 'conv2d' else 1,
            layer_type=layer_type,
            offset=1,
            figsize=(0.65*nvars*n_cols, 0.65*nvars*n_rows),
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            conv2d_cmap_name=conv2d_cmap_name,
            fc_color_name=fc_color_name,
            show_colorbar=show_colorbar,
            rescale_blocks=rescale_blocks
        )
        
def get_min_max_values(a):
    return a.min(), a.max()    
        
def shp_histograms_compact_plot(
    shpv_path,
    shptv_path,
    network_module_names,
    variable_names,
    shp_values_func=None,
    figsize=(10, 3),
    eps=1e-20,
    zero_atol=1e-10
):
    shpv_vn, shptv_vn = variable_names
    for i_mn, module_name in enumerate(network_module_names):
        shptv_array, act_shape_tv = get_values(shptv_path, module_name, shptv_vn, values_func=None)
        #with h5py.File(shptv_path, 'r') as tv:
        #    shptv_array = tv[f'{module_name}/{shptv_vn}'][:]
        #    #act_shape_tv = tv.attrs[f'{module_name}']
        #    act_shape_tv = tuple(tv[f'{module_name}'].attrs['shape'])
        shpv_array, act_shape_shpv = get_values(shpv_path, module_name, shpv_vn, shp_values_func)
        #with h5py.File(shpv_path, 'r') as shpv:
        #    shpv_array = shpv[f'{module_name}/{shpv_vn}'][:]
        #    #act_shape_shpv = shpv.attrs[f'{module_name}']
        #    act_shape_shpv = tuple(shpv[f'{module_name}'].attrs['shape'])
        shpvtv_estimate = shpv_array.sum(axis=0, keepdims=True)
        #shpvtv_estimate = shpv_array.sum(axis=0)
        normalized_shpv_array = sensitivity_analysis.shapley.normalize_shapley(
            shpv_array, shpvtv_estimate, eps, zero_atol, carefully=True, clip_negative=True
        )
        print(
            module_name,
            get_min_max_values(shptv_array),
            get_min_max_values(shpvtv_estimate),
            get_min_max_values(shpv_array),
            get_min_max_values(normalized_shpv_array),
        )
        plot_histograms(
            value_arrays=[
                shptv_array,
                shpvtv_estimate,
                shpv_array,
                normalized_shpv_array
            ],
            value_names=[
                'Total variances (dist.)',
                'Total variances (sum.)',
                'Shapley values (not normalized)',
                'Shapley values'
            ],
            figsize=figsize
        )

def si_histograms_compact_plot(
    si_path,
    sitv_path,
    value_indices,
    network_module_names,
    values_func=None,
    variable_names=(('si+sT', 'si2'), 'sitv'),
    figsize=(10, 3),
):
    si_vn, sitv_vn = variable_names
    (si_ind, sT_ind), si2_ind = value_indices
    for i_mn, module_name in enumerate(network_module_names):
        with h5py.File(sitv_path, 'r') as sitv:
            sitv_array = sitv[f'{module_name}/{sitv_vn}'][:]
        with h5py.File(si_path, 'r') as si:
            sisT_array = si[f'{module_name}/{si_vn[0]}'][:]
            si2_array = si[f'{module_name}/{si_vn[1]}'][:]
        print(
            module_name,
            get_min_max_values(sitv_array),
            get_min_max_values(sisT_array[si_ind]),
            get_min_max_values(sisT_array[sT_ind]),
            get_min_max_values(si2_array[si2_ind])
        )
        if values_func is not None:
            sitv_array, sisT_array[si_ind], sisT_array[sT_ind], si2_array[si2_ind] = values_func(
                sitv_array, sisT_array[si_ind], sisT_array[sT_ind], si2_array[si2_ind]
            )
        plot_histograms(
            value_arrays=[
                sitv_array,
                sisT_array[si_ind],
                sisT_array[sT_ind],
                si2_array[si2_ind]
            ],
            value_names=[
                'Total variances',
                r'$S_i$',
                r'$S_i^T$',
                r'$S_{ij}$'
            ],
            figsize=figsize
        )

def get_values(val_path, module_name, values_name, values_func=None):
    with h5py.File(val_path, 'r') as vals:
        current_group = vals[module_name]
        if isinstance(values_name, str):
            current_values = current_group[values_name][:]
            current_values = np.array(current_values)
        else:
            current_values = []
            for vname in values_name:
                current_values.append(current_group[vname][:])
        
        act_shape = tuple(current_group.attrs['shape'])
        #values = fval[f'{module_name}/{values_name}'][:]
        #act_shape = tuple(fval[f'{module_name}'].attrs['shape'])
    if values_func is not None:
        current_values = values_func(current_values)
    return current_values, act_shape

def get_shp_values(shpv_path, module_name, values_name, normalize, values_func=None, eps=1e-20):
    values, act_shape = get_values(shpv_path, module_name, values_name, values_func)
    if normalize:
        values = sensitivity_analysis.shapley.normalize_shapley(
            values, eps=eps, carefully=True, clip_negative=True
        )
        #values = values.reshape((values.shape[0], -1))
        #tv_estimate = values.sum(axis=0, keepdims=True)
        #values /= eps+tv_estimate
    return values, act_shape
    
        
def shpv_compact_plot(
    shpv_path,
    network_module_names,
    values_name,
    variable_names,
    n_conv_modules,
    n_cols,
    n_col_blocks,
    n_rows_fc=None,
    vmin=0,
    vmax=1,
    normalize=True,
    eps=1e-20,
    figsize=None,
    values_func=None,
    save_filename_base=None,
    save_dirname=None, 
    conv2d_cmap_name='Reds',
    fc_color_name='red',
    show=True,
    show_colorbar=True,
    global_colorbounds=False,
    global_bounds_conv_only=True,
):
    save_path = None
    if global_colorbounds:
        #assert (vmin is None) and (vmax is None)
        if vmin is None:
            vmin = np.inf # vmin = vmin or np.inf, but sometimes better write it explicitly
        if vmax is None:
            vmax = -np.inf
        for i_mn, module_name in enumerate(network_module_names):
            if global_bounds_conv_only and (i_mn == n_conv_modules):
                break
            values, act_shape = get_shp_values(shpv_path, module_name, values_name, normalize, values_func, eps)
            vmin = min(vmin, values.min())
            vmax = max(vmax, values.max())
    
    for i_mn, module_name in enumerate(network_module_names):
        
        values, act_shape = get_shp_values(shpv_path, module_name, values_name, normalize, values_func, eps)
        print(module_name)
        if i_mn < n_conv_modules:
            n_rows = act_shape[0] // n_cols
            values = values.reshape((-1, ) + act_shape)#[None]
            #values = values[:, :, :, None, :, :]
            values = values[:, :, None, :, :]
            layer_type = 'conv2d'
        else:
            n_rows = n_rows_fc or max(1, act_shape[0] // (4*n_cols))
            values = values.reshape((-1, ) + act_shape)#[None]
            layer_type = 'fc'
            if global_bounds_conv_only:
                vmin = vmax = None
        if save_dirname is not None:
            save_filename = f'{save_filename_base}_{module_name}.pdf'
            save_path = os.path.join(save_dirname, save_filename)
        plot_many_block_images_scaled(
            values,
            block_names=variable_names,
            n_rows=n_rows,
            n_cols=n_cols,
            n_col_blocks=n_col_blocks,
            masks=None,
            layer_type=layer_type,
            offset=1,
            figsize=figsize,
            vmin=vmin,
            vmax=vmax,
            save_path=save_path,
            show=show,
            conv2d_cmap_name=conv2d_cmap_name,
            fc_color_name=fc_color_name,
            rescale_blocks=True,
            show_colorbar=show_colorbar,
            colorbar_path=os.path.join(save_dirname, f'{save_filename_base}_{module_name}_colorbar.pdf'),
        )

def plot_colorbar(figsize=None, w=1., h=1., vmin=0, vmax=1, dv=0.2, ticks=None, label='', cmap=None):
    #fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig = plt.figure(figsize=figsize)
    # [left, bottom, width, height] # ax.insert_axes(..., transform=ax.transAxes)
    #img = ax.imshow(np.array([[vmin, vmax]]), cmap="Oranges")
    #img.set_visible(False)
    
    ax = fig.add_axes([0., 0., w, h])
    #ax.set_clim([vmin, vmax])
    if ticks is None:
        #base = 5
        #lval = base*np.floor(vmin / base)
        #rval = base*np.ceil(vmax / base)
        ticks = np.arange(vmin, vmax+dv, dv)
        #ticks = np.linspace(lval, rval, base+1)
    cbar = mpl.colorbar.ColorbarBase(
        ax=ax,
        cmap=plt.get_cmap(cmap),
        norm=mpl.colors.Normalize(vmin, vmax),
        alpha=None,
        values=None,
        boundaries=None,
        orientation='horizontal',
        ticklocation='top',
        extend='neither',
        spacing='uniform',
        ticks=ticks,
        format=None,
        drawedges=False,
        filled=True,
        extendfrac=None,
        extendrect=False,
        label=label,
        #cmap=mpl.colors.Colormap(cmap)
    )
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels(np.round(ticks, 2))
    #cbar.ax.xaxis.set_ticks_position('top')
    return fig
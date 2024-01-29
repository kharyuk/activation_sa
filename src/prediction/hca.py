import functools
import itertools
import warnings

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import sklearn.cluster
import scipy.cluster.hierarchy

import preparation.visualize
import preparation.single_unit

from . import utils


_good_looking_cmap_list = [
    'afmhot_r',
    'autumn_r',
    'binary',
    'bone_r',
    'brg_r',
    'cividis_r',
    'coolwarm',
    'coolwarm_r',
    'copper_r',
    'cubehelix_r',
    'gist_earth_r',
    'gist_gray_r',
    'gist_heat_r',
    'gist_ncar_r',
    'gist_rainbow_r',    
    'gist_stern_r',
    'gist_yarg',
    'gnuplot2_r',
    'gnuplot_r',
    'gray_r',    
    'hot_r',
    'inferno_r',
    'jet',
    'jet_r',
    'magma_r',
    'nipy_spectral_r',
    'ocean_r',
    'pink_r',
    'plasma_r',
    'rainbow',
    'rainbow_r',
    'seismic',
    'seismic_r',
    'summer_r',
    'tab10_r',
    'tab20b_r',
    'terrain_r',
    'turbo',
    'turbo_r',
    'twilight_shifted',
    'twilight_shifted_r',
    'viridis_r',
    'Blues',
    'BrBG',
    'BrBG_r',
    'BuGn',
    'BuPu',
    'CMRmap_r',
    'Dark2',
    'GnBu',
    'Greens',
    'Greys',
    'OrRd',
    'Oranges',
    'PRGn',
    'PRGn_r',
    'PiYG',
    'PiYG_r',
    'PuBu',
    'PuBuGn',
    'PuOr',
    'PuOr_r',
    'PuRd',
    'Purples',
    'RdBu',
    'RdBu_r',
    'RdGy',
    'RdGy_r',
    'RdPu',
    'RdYlBu',
    'RdYlBu_r',
    'RdYlGn',
    'RdYlGn_r',
    'Reds',
    'Spectral',
    'Spectral_r',
    'YlGn',
    'YlGnBu',
    'YlOrBr',
    'YlOrRd',
]



# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, max_d=0.025, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram = scipy.cluster.hierarchy.dendrogram(linkage_matrix, **kwargs)

    clusts = scipy.cluster.hierarchy.fcluster(linkage_matrix, max_d, criterion='distance')
    #print(clusts)
    
    return dendrogram, clusts



def get_colors(n_colors, cmap='rainbow_r'):
    colors = [
        mpl.colors.rgb2hex(rgb[:3])
        for rgb in getattr(plt.cm, cmap)(np.log10(np.linspace(1, 10, n_colors)))
        #for rgb in getattr(plt.cm, cmap)(np.exp(np.linspace(-1e10, 0, n_colors)))
    ]
    return colors

def link_color_func(x, colors):
    return colors[x]

def plot_hca_spearman_dendrogram(
    featured_measurements_dict,
    augmentation_set_number,
    extract_auxilliary_names=False,
    linkage='average',
    figsize=(14, 14),
    show=True,
    cmap='OrRd',
    label_cmap='jet',
    max_d_label=0.025,
    label_background_color='white',
    label_fontsize=10,
    save_path=None,
):
    augmentation_names, augmentation_and_auxilliary_names = utils.get_shortened_variable_names_single_augset(
        augmentation_set_number,
        extract_auxilliary_names=extract_auxilliary_names,
    )
        
    n_augs = len(augmentation_names)
    n_augs_aux = len(augmentation_and_auxilliary_names)
    for i, M in enumerate(featured_measurements_dict.values()):
        if i == 0:
            n_features = M.shape[-1]
        assert M.shape == (n_augs, n_augs_aux, n_features)
    n_values = len(featured_measurements_dict)
    
    n_clusters = n_augs*n_augs_aux
    n_colors = 2*n_clusters - 1
    colors = get_colors(n_colors, cmap=cmap)
    
    labels = list(
        map(
            lambda x: f'{x[0]}--{x[1]}',
            itertools.product(augmentation_names, augmentation_and_auxilliary_names)
        )
    )
    
    fig, ax = plt.subplots(1, n_values, figsize=figsize)
    for i_val, (value_name, values) in enumerate(featured_measurements_dict.items()):
        cor_mat, _ = scipy.stats.spearmanr(
            values.reshape((-1, n_features)), axis=1
        )
        hca = sklearn.cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            #metric='precomputed',
            compute_full_tree=True,
            linkage=linkage,
            distance_threshold=None,
            compute_distances=True,
        )
        cor_mat = np.sqrt(1-np.abs(cor_mat))
        hca.fit(cor_mat)

        cax = ax[i_val] if n_values > 1 else ax
        dendrogram, clusts = plot_dendrogram(
            hca,
            max_d_label,
            ax=cax,
            labels=labels,
            leaf_font_size=12.,
            orientation='right',
            color_threshold=None,
            link_color_func=functools.partial(link_color_func, colors=colors),
        )
        cax.axvline(x=max_d_label, color='black', ls=':')
        #margin = 0#.005
        #cax.set_xlim((-margin+(1.-cor_mat.max()), (1.-cor_mat.min())+margin))
        cax.set_xlabel(r'$\sqrt{1 - |\rho_{\mathrm{Spearman}}|}$')#Spearman correlation')
        #xticks = cax.get_xticks()
        #xtick_labels = cax.get_xticklabels()
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')
        #    cax.set_xticklabels(1. - xticks)
        cax.grid(alpha=0.65)
        cax.set_title(f'({value_name})')
        '''
        # https://stackoverflow.com/questions/66399501/how-to-draw-colored-rectangles-around-grouped-clusters-in-dendogram
        for col in cax.collections[:1]:
            ymin, ymax = np.inf, -np.inf
            xmin, xmax = np.inf, -np.inf
            for p in col.get_paths():
                box = p.get_extents()
                (x0, y0), (x1, y1) = box.get_points()
                xmin, ymin = min(xmin, x0), min(ymin, y0)
                xmax, ymax = max(xmax, x1), max(ymax, y1)
            rectangle = plt.Rectangle(
                xy=(-1, ymin),
                width=xmax-xmin,
                height=ymax-ymin,
                facecolor=col.get_color()[0],
                alpha=0.1,
                edgecolor='none',
            )
            cax.add_patch(rectangle)
        '''
        
        label_colors = get_colors(len(set(clusts))+1, cmap=label_cmap)
        y_labels = cax.get_ymajorticklabels()
        for c_y in y_labels:
            clabel = c_y.get_text()
            ind = labels.index(clabel)
            col_ind = clusts[ind]
            c_y.set_color(link_color_func(col_ind, label_colors))
            #c_y.set_facecolor('green')
            c_y.set_backgroundcolor(label_background_color)
            c_y.set_fontsize(label_fontsize)
            #c_y.set_alpha(0.5)

    
    #margin = 0.1
    #fig.subplots_adjust(left=margin, right=1-margin)
    plt.tight_layout(pad=0.15)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.clf()
    


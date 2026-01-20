import PIL
import matplotlib.pyplot as plt

import os
import time
from IPython.display import clear_output

import torch
import numpy as np

def visually_inspect_image_transform(transformer, image, n_samples=5):
    nrows, ncols = 2, n_samples
    figsize_horizontal = int(round(1.5*ncols))
    figsize_vertical = 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(figsize_horizontal, figsize_vertical))

    a = np.transpose( np.array(image), (1, 2, 0))
    #im = PIL.Image.fromarray(a)
    im = np.array(image)
    im = im/im.max()

    for i in range(ncols):
        ax[0, i].imshow(a)
        ax[0, i].axes.xaxis.set_ticklabels([])
        ax[0, i].axes.yaxis.set_ticklabels([])
        ax[0, i].axes.xaxis.set_ticks([])
        ax[0, i].axes.yaxis.set_ticks([])
        
        #try:
        #t_im = transformer(im)
        #except:
        t_im = transformer(torch.Tensor(im))
        t_im = np.transpose( np.array(t_im), (1, 2, 0))
        if t_im.max() > 1:
            t_im = t_im.astype('i')
        #print(np.max(t_im/255 - a/255))
        #print(t_im.max(), t_im.min())
        ax[1, i].imshow(np.array(t_im))
        ax[1, i].axes.xaxis.set_ticklabels([])
        ax[1, i].axes.yaxis.set_ticklabels([])
        ax[1, i].axes.xaxis.set_ticks([])
        ax[1, i].axes.yaxis.set_ticks([])
        
        if i == 0:
            ax[0, i].set_ylabel('Original')
            ax[1, i].set_ylabel('Transformed')
        
    plt.tight_layout()
    plt.show()
    
def dynamic_multi_plot(dataset, wait_interval, n_cols, figsize, output_transform=None):
    plt.clf()
    clear_output()

    history = []

    for (x, y, z) in dataset:
        #x, y = xy
        if output_transform is not None:
            x = output_transform(x)
        history.append(
            (
                x.detach().numpy().transpose([1, 2, 0]),
                dataset.dataset[0].dataset.classes[y]
            )
        )
        if len(history) > n_cols:
            del history[0]
        fig, ax = plt.subplots(1, n_cols, figsize=figsize)
        for i in range(n_cols):
            if len(history) < n_cols:
                ind = n_cols-i-1
            else:
                ind = i
            if i < len(history):
                ax[ind].imshow(history[i][0])
                ax[ind].set_title(history[i][1])
            else:
                ax[ind].imshow(0.+np.ones_like(history[0][0]))
            ax[ind].tick_params(
                top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False
            )
        plt.show()
        try:
            time.sleep(wait_interval)
            clear_output()
        except KeyboardInterrupt:
            break

def sample_image_transform(
    transformer, image, save_path, n_samples=5, figsize_vertical=5
):
    n_samples = min(n_samples, 99999)
    figsize_horizontal = figsize_vertical

    a = np.transpose( np.array(image), (1, 2, 0))
    #im = PIL.Image.fromarray(a)
    im = np.array(image)
    im = im/im.max()

    for i in range(n_samples+1):
        fig, ax = plt.subplots(1, 1, figsize=(figsize_horizontal, figsize_vertical), frameon=False)

        if i == 0:
            ax.imshow(a, extent=[0, 1, 0, 1])
        else:
            t_im = transformer(torch.Tensor(im))
            t_im = np.transpose( np.array(t_im), (1, 2, 0))
            if t_im.max() > 1:
                t_im = t_im.astype('i')
            ax.imshow(np.array(t_im), extent=[0, 1, 0, 1])
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines[:].set_visible(False)

        current_path = os.path.join(save_path, f"{i:05d}.jpg")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        plt.savefig(current_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
        
def save_dynamic_multi_plot(dataset, n_samples, save_path, figsize, output_transform=None):
    n_samples = min(n_samples, 99999)
    for i, (x, y, z) in enumerate(dataset):
        if i == n_samples:
            break
        #x, y = xy
        if output_transform is not None:
            x = output_transform(x)
        cx = x.detach().numpy().transpose([1, 2, 0]),
        fig, ax = plt.subplots(1, 1, figsize=figsize, frameon=False)
        
        ax.imshow(np.array(cx[0]), extent=[0, 1, 0, 1])
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines[:].set_visible(False)
        current_path = os.path.join(save_path, f"{i:05d}.jpg")
        #plt.axis('off')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        #plt.tight_layout(pad=0)
        plt.savefig(current_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
        
        
        
    
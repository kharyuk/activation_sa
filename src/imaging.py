import matplotlib.pyplot as plt
import numpy as np

def normalize(images, alpha=2, beta=1.):
    return alpha*images - beta

def inverse_normalize(images, alpha=2, beta=1):
    return (images+beta)/alpha

def plot_images(images, n_rows, n_cols, figsize=None):
    batch_size = images.shape[0]
    offset = batch_size // (n_rows*n_cols)
    assert offset > 0
    if figsize is None:
        figsize = (1.5*n_rows, 1.5*n_cols)

    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=figsize, gridspec_kw={'wspace': 0, 'hspace': 0}
    )
    ind = 0
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].imshow(np.transpose(images[ind], (1, 2, 0)))
            ax[i, j].axes.xaxis.set_ticklabels([])
            ax[i, j].axes.yaxis.set_ticklabels([])
            ax[i, j].axes.xaxis.set_ticks([])
            ax[i, j].axes.yaxis.set_ticks([])
            ind += offset
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(
    ax_i, confusion_matrix, class_names, title='', cmap='gist_heat',
    interpolation='none', x_rotation=30
):
    num_classes = len(class_names)
    ticks = np.arange(num_classes)
    ax_i.matshow(confusion_matrix, cmap=cmap, interpolation=interpolation)
    ax_i.set_xticks(ticks)
    ax_i.set_xticklabels(class_names, rotation=x_rotation)
    ax_i.set_yticks(ticks)
    ax_i.set_yticklabels(class_names, rotation=x_rotation)
    ax_i.set_title(title)
    
def plot_confusion_matrices(cms, subtitles, classes, title=None, figsize=(12, 5), fontsize=20):
    N = len(cms)
    assert N == len(subtitles)
    if title is None:
        title = 'Confusion matrices (top-1)'
    fig, ax = plt.subplots(1, N, figsize=figsize)
    fig.suptitle(title, fontsize=fontsize)
    for i in range(N):
        plot_confusion_matrix(
            ax[i], cms[i], classes, title=subtitles[i],
        )

    plt.tight_layout()
    plt.show()
    
def massive_plot_cms(cms, classes, augmentaton_sets_names):
    current_cms = [cms[x] for x in ['train', 'valid', 'test']]
        
    imaging.plot_confusion_matrices(
        current_cms, subtitles=('train', 'valid', 'test'), classes=classes
    )
    subtitles = ['regular'] + augmentaton_sets_names
    
    name = 'valid'
    names = list(map(lambda x: name if len(x)==0 else f'{name}+{x}', ['']+augmentaton_sets_names))
    current_cms = [cms[x] for x in names]
    imaging.plot_confusion_matrices(
        current_cms, subtitles=subtitles, classes=classes,
        title='Confusion matrices (top1), valid'
    )
    name = 'test'
    names = list(map(lambda x: name if len(x)==0 else f'{name}+{x}', ['']+augmentaton_sets_names))
    current_cms = [cms[x] for x in names]
    imaging.plot_confusion_matrices(
        current_cms, subtitles=subtitles, classes=classes,
        title='Confusion matrices (top1), test'
    )
    
def plot_learning_curve(
    train_loss_values_list,
    valid_loss_values_list,
    n_epoch,
    validate_every_nepoch=1,
    figsize=(10, 8),
    show=True
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.semilogy(train_loss_values_list, '*-', color='green', linewidth=1, label='train')
    ax.semilogy(
        validate_every_nepoch+np.arange(0, n_epoch, validate_every_nepoch),
        valid_loss_values_list,
        '*-',
        color='red',
        linewidth=1,
        label='valid'
    )
    ax.set_title('Learning curves')
    ax.set_xlabel('Num. epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend()
    ax.grid(alpha=0.5)
    if show:
        plt.show()
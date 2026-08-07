import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def generate_legend(
    augmentation_set_numbers_list,
    colors,
    figsize=(5, 1)
):
    handlers = [
        Patch(facecolor=colors[i], label=f"Aug.set {augmentation_set_numbers_list[i]}")
        for i in range(len(augmentation_set_numbers_list))
    ]
    labels = [hand.get_label() for hand in handlers]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.legend(
        handlers, labels, frameon=True,
        ncol=len(augmentation_set_numbers_list)
    )
    return fig, ax

def plot_first_sobol_outs(
    x_si,
    y_si,
    cnt_si,
    network_modules,
    augmentation_set_numbers_list,
    colors,
    ymin=1e-8,
    ymax=1e1,
    xmin=1e-5,
    xmax=1e0,
    figsize=(5, 3),
    title_size=10,
    n_rows=2
):
    n_cols = 1 + (len(network_modules)-1) // n_rows
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=figsize)
    for k, module_name in enumerate(network_modules):
        i = k // n_cols
        j = k % n_cols
        for i_aug, aug_set in enumerate(augmentation_set_numbers_list):
            ax[i, j].loglog(
                abs(x_si), y_si[module_name][aug_set],
                label=f"aug={aug_set}", color=colors[i_aug], alpha=0.7
            )
            ax[i, j].loglog(
                abs(x_si),
                cnt_si[module_name][aug_set],
                "--", lw=1, color=colors[i_aug], #label=f"aug={aug_set}",
            )
        #ax[i, j].axhline(1., xmin, xmax, ls='--', color='darkgray', alpha=0.75)
        ax[i, j].axvline(1e-2, ymin, ymax, ls='-', color='black', alpha=1., lw=1., zorder=0)
        ax[i, j].set_title(f"{module_name}", size=title_size)
        ax[i, j].set_xlim(xmin, xmax)
        ax[i, j].set_ylim(ymin, ymax)
        ax[i, j].set_xticks((1e-5, 1e-4, 1e-3, 1e-2, 1e-1))
        ax[i, j].set_xticklabels((r"$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$"))
        ax[i, j].set_yticks(
            #(1e-7, 1e-6,
            (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1)
            #, 1e2, 1e3))
        )
        ax[i, j].set_yticklabels(
            #(r"$10^{-7}$", "", 
            ("$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$", "", r"$10^{1}$")
            #, "", r"$10^{3}$")
        )
        #ax[i, j].legend()
        ax[i, j].grid(alpha=0.5)
        if j == 0 and i == 0:
            #ax[i, j].set_ylabel("Max.variance")# value")
            fig.text(0, 0.5, "Contrast variation index (-) / ratio (--)", va="center", rotation=90)
        if i == n_rows-1 and j == 0:#(j % 2) == 0:
            #ax[i, j].set_xlabel(r"$\Delta_{0^{-}}$ (first-order)")
            fig.text(0.5, -0.05, r"$\Delta_{0^{-}}$ (first-order Sobol indices)", ha="center")
    j += 1
    while j < n_cols:
        ax[i-1, j].xaxis.set_tick_params(labelbottom=True)
        j += 1
    Nplots = len(network_modules)
    for k in range(Nplots, n_rows*n_cols):
        i = k // n_cols
        j = k % n_cols
        ax[i, j].axis("off")
    return fig, ax

def plot_total_sobol_outs(
    x_sT,
    y_sT,
    cnt_sT,
    network_modules,
    augmentation_set_numbers_list,
    colors,
    ymin=1e-8,
    ymax=1e1,
    xmin=1e-5,
    xmax=1e0,
    figsize=(5, 3),
    title_size=10,
    n_rows=2
):
    n_cols = 1 + (len(network_modules)-1) // n_rows
    
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=figsize)
    for k, module_name in enumerate(network_modules):
        i = k // n_cols
        j = k % n_cols
        for i_aug, aug_set in enumerate(augmentation_set_numbers_list):
            ax[i, j].loglog(
                abs(x_sT - 1.), y_sT[module_name][aug_set],
                label=f"aug={aug_set}", color=colors[i_aug], alpha=0.7
            )
            ax[i, j].loglog(
                abs(x_sT - 1.),
                cnt_sT[module_name][aug_set],
                "--", lw=1, color=colors[i_aug], #label=f"aug={aug_set}",
            )
        #ax[i, j].axhline(1., xmin, xmax, ls='--', color='darkgray', alpha=0.75)
        ax[i, j].axvline(1e-2, ymin, ymax, ls='-', color='black', alpha=1., lw=1., zorder=0)
        ax[i, j].set_title(f"{module_name}", size=title_size)
        ax[i, j].set_xlim(xmin, xmax)
        ax[i, j].set_ylim(ymin, ymax)
        if j == 0 and i == 0:
            fig.text(0, 0.5, "Contrast variation index (-) / ratio (--)", va="center", rotation=90)
        if i == n_rows-1 and j == 0:
            fig.text(0.5, -0.05, r"$\Delta_{1^{+}}$ (total Sobol indices)", ha="center")
        ax[i, j].set_xticks((1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0))
        ax[i, j].set_xticklabels((r"$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$", ""))
        ax[i, j].set_yticks(
            #(1e-7, 1e-6,
            (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1)
            #, 1e2, 1e3))
        )
        ax[i, j].set_yticklabels(
            #(r"$10^{-7}$", "", 
            ("$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$", "", r"$10^{1}$")
            #, "", r"$10^{3}$")
        )
        ax[i, j].grid(alpha=0.5)
        #ax[i, j].legend()
    j += 1
    while j < n_cols:
        ax[i-1, j].xaxis.set_tick_params(labelbottom=True)
        j += 1
        
    Nplots = len(network_modules)
    for k in range(Nplots, n_rows*n_cols):
        i = k // n_cols
        j = k % n_cols
        ax[i, j].axis("off")
    return fig, ax

def plot_shapley_outs(
    x_shpv,
    y_shpv,
    cnt_shpv,
    network_modules,
    augmentation_set_numbers_list,
    colors,
    ymin=1e-5,
    ymax=1e1,
    xmin=1e-5,
    xmax=1e2,
    figsize=(5, 3),
    title_size=10,
    n_rows=2
):
    n_cols = 1 + (len(network_modules)-1) // n_rows
    
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=figsize)
    for k, module_name in enumerate(network_modules):
        i = k // n_cols
        j = k % n_cols
        for i_aug, aug_set in enumerate(augmentation_set_numbers_list):
            ax[i, j].loglog(
                abs(x_shpv), y_shpv[module_name][aug_set], label=f"aug={aug_set}",
                color=colors[i_aug], alpha=0.7
            )
            ax[i, j].loglog(
                abs(x_shpv),
                #uvf_shpv[module_name][aug_set],
                cnt_shpv[module_name][aug_set],
                "--", lw=1, color=colors[i_aug], #label=f"aug={aug_set}",
            )
        #ax[i, j].axhline(1., xmin, xmax, color='darkgray', alpha=1, lw=0.95)
        ax[i, j].axvline(1e-2, ymin, ymax, ls='-', color='black', alpha=1., lw=1., zorder=0)
        ax[i, j].set_title(f"{module_name}", size=title_size)
        ax[i, j].set_ylim(ymin, ymax)
        ax[i, j].set_xlim(xmin, xmax)
        if j == 0 and i == 0:
            fig.text(0, 0.5, "Contrast variation index (-) / ratio (--)", va="center", rotation=90)
        if i == n_rows-1 and j == 0:
            fig.text(0.5, -0.05, r"$\Delta_{0^{-}}$ (Shapley values)", ha="center")
        ax[i, j].set_xticks((1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1))
        ax[i, j].set_xticklabels((r"$10^{-4}$", "", r"$10^{-2}$", "", r"$10^{0}$", ""), size=7.5)
        ax[i, j].set_yticks((1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1))#, 1e2, 1e3))
        ax[i, j].set_yticklabels(
            ("$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$", "", r"$10^{1}$")#, "", r"$10^{3}$")
        )
        ax[i, j].grid(alpha=0.5)
    j += 1
    while j < n_cols:
        ax[i-1, j].xaxis.set_tick_params(labelbottom=True)
        j += 1
    Nplots = len(network_modules)
    for k in range(Nplots, n_rows*n_cols):
        i = k // n_cols
        j = k % n_cols
        ax[i, j].axis("off")
    return fig, ax
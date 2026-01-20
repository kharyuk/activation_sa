from typing import List
from typing import Union
import os
import math
import numpy as np
import scipy.stats

def jaccard_index(a: np.ndarray, b: np.ndarray) -> float:
    nom: int = np.intersect1d(a, b).size
    den: int = np.union1d(a, b).size
    return nom / den

def two_boxes_full_simulation(
    N: int,
    s: int,
    n: int,
    n_trials: int = 100_000,
    verbose: bool = True
) -> List[float]:
    
    mean_jac_vals_list: List[float] = []
    n_samples: int = N*s
    
    trial: int
    for trial in range(n_trials):
        ref_sample: np.ndarray = np.random.choice(
            N, size=n, replace=False
        )
        vals: List[float] = []
        k: int
        for k in range(n_samples):
            cur_sample: np.ndarray = np.random.choice(
                N, size=n, replace=False
            )
            cval: float = jaccard_index(cur_sample, ref_sample)
            vals.append(cval)
        cur_mean: float = np.mean(vals)
        mean_jac_vals_list.append(cur_mean)
        if verbose:
            print(
                f"{trial+1}/{n_trials}: "
                f" cmean={np.mean(mean_jac_vals_list):.5f}"
                f" cstd={np.std(mean_jac_vals_list):.5f}",
                end="\r"
            )
    return mean_jac_vals_list

def compute_proba(N: int, n: int, k: int) -> float:
    alpha: float = math.comb(n, k)
    beta: float = math.comb(N-n, n-k)
    gamma: float = math.comb(N, n)
    return alpha*beta/gamma 

def compute_jac_outcome(k: int, n: int) -> float:
    return k/(2*n-k)

def two_boxes_distribution_sampling(
    N: int,
    s: int,
    n: int,
    n_trials: int = 100_000,
    verbose: bool = True,
    verbose_n_steps: int = 1000,
    verbose_stats: bool = False,
    save_path: Union[str, None] = None,
    save_n_steps: int = 1000
) -> np.ndarray:
    mj_values: np.ndarray = np.zeros((n_trials, ), dtype='d')
    n_samples: int = N*s

    rand_variable: scipy.stats._distn_infrastructure.rv_frozen = (
        scipy.stats.hypergeom(N, n, n)
    )
    k_values: np.ndarray = np.arange(n+1)
    pmf_values: np.ndarray = rand_variable.pmf(k_values)

    if save_path is not None:
        save_filename: str = (
            f"mj_values_N={N}_n={n}_s={s}_trials={n_trials}"
        )
        save_path: str = os.path.join(
            save_path, save_filename
        )
        if save_path.endswith("/"):
            save_path = save_path[:-1]

    k_trial: int
    for k_trial in range(n_trials):
        k_samples: np.ndarray = np.random.choice(
            k_values, size=n_samples, p=pmf_values
        )
        j_values: np.ndarray = compute_jac_outcome(k_samples, n)
        cur_mean: float = np.mean(j_values)
        mj_values[k_trial] = cur_mean
        if (
            verbose and (
                ((k_trial+1) % verbose_n_steps) == 0
                or k_trial == (n_trials-1)
            )
        ):
            print(
                f"{k_trial+1}/{n_trials}: ", end=''
            )
            if not verbose_stats:
                print(end="\r")
            else:
                print(
                    f" cmean={np.mean(mj_values[:k_trial+1]):.5f}"
                    f" cstd={np.std(mj_values[:k_trial+1]):.5f}",
                    end="\r"
                )
        if (
            (save_path is not None) and (
                ((k_trial+1) % save_n_steps) == 0
                or k_trial == (n_trials-1)
            )
        ):
            np.savez_compressed(
                f"{save_path}_{k_trial+1}.npz", mj_values=mj_values,
                N=N, n=n, s=s, n_trials=n_trials
            )
    return mj_values

def compute_hiqr_outlier_threshold(
    values: List[float]
) -> float:
    q1: float = np.quantile(values, q=0.25)
    q3: float = np.quantile(values, q=0.75)
    iqr: float = q3 - q1
    return 1.5*iqr + q3

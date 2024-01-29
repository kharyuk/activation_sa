from scipy.stats import norm

import numpy as np
import pandas as pd

from SALib.analyze.sobol import (
    to_df, Si_to_pandas_dict, Si_list_to_dict,
    create_task_list
)

from SALib.analyze import common_args
from SALib.util import (
    read_param_file, compute_groups_matrix, ResultDict,
    extract_group_names, _check_groups
)
from types import MethodType

from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import combinations, zip_longest


def safe_division(nominator, denominator, eps=1e-8, atol=1e-8, axis=None):
    #print(nominator.shape, denominator.shape)
    ind0 = np.where(np.isclose(nominator, 0., atol=atol))
    result = nominator.copy()
    result[ind0] = 0.
    if axis is None:
        ind_nnz = np.where(~np.isclose(nominator, 0., atol=atol))
        result[ind_nnz] = result[ind_nnz] / (eps + denominator[ind_nnz])
    else:
        if axis != 0:
            result = np.swapaxes(result, 0, axis)
            denominator = np.swapaxes(denominator, 0, axis)
        for i in range(len(result)):
            ind_nnz = np.where(~np.isclose(nominator[i], 0., atol=atol))
            result[i][ind_nnz] = result[i][ind_nnz] / (eps + denominator[0][ind_nnz])
        if axis != 0:
            result = np.swapaxes(result, 0, axis)
    return result

def analyze(problem, Y, calc_second_order=True, num_resamples=100,
            conf_level=0.95, print_to_console=False, parallel=False,
            n_processors=None, keep_resamples=False, seed=None, eps=1e-7):
    """Perform Sobol Analysis on model outputs.

    Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf', where
    each entry is a list of size D (the number of parameters) containing the
    indices in the same order as the parameter file.  If calc_second_order is
    True, the dictionary also contains keys 'S2' and 'S2_conf'.

    Compatible with
    ---------------
    * `saltelli`

    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    num_resamples : int
        The number of resamples (default 100)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)
    keep_resamples : bool
        Whether or not to store intermediate resampling results (default False)

    References
    ----------
    .. [1] Sobol, I. M. (2001).  "Global sensitivity indices for nonlinear
           mathematical models and their Monte Carlo estimates."  Mathematics
           and Computers in Simulation, 55(1-3):271-280,
           doi:10.1016/S0378-4754(00)00270-6.
    .. [2] Saltelli, A. (2002).  "Making best use of model evaluations to
           compute sensitivity indices."  Computer Physics Communications,
           145(2):280-297, doi:10.1016/S0010-4655(02)00280-1.
    .. [3] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
           S. Tarantola (2010).  "Variance based sensitivity analysis of model
           output.  Design and estimator for the total sensitivity index."
           Computer Physics Communications, 181(2):259-270,
           doi:10.1016/j.cpc.2009.09.018.

    Examples
    --------
    >>> X = saltelli.sample(problem, 512)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=True)

    """
    if seed:
        # Set seed to ensure CIs are the same
        rng = np.random.default_rng(seed).integers
    else:
        rng = np.random.randint

    # determining if groups are defined and adjusting the number
    # of rows in the cross-sampled matrix accordingly
    groups = _check_groups(problem)
    if not groups:
        D = problem['num_vars']
    else:
        _, D = extract_group_names(groups)
        
    sizeY, M = Y.shape

    if calc_second_order and Y.size % (2 * D + 2) == 0:
        N = int(sizeY / (2 * D + 2))
    elif not calc_second_order and sizeY % (D + 2) == 0:
        N = int(sizeY / (D + 2))
    else:
        raise RuntimeError("""
        Incorrect number of samples in model output file.
        Confirm that calc_second_order matches option used during sampling.""")

    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    # normalize the model output
    #print('\n start')
    Y = safe_division(
        Y - Y.mean(axis=0, keepdims=True), Y.std(axis=0, keepdims=True),
        eps=eps, atol=eps*0.1, axis=0
    )
    #print('\n Normalized')

    A, B, AB, BA = separate_output_values(Y, D, N, calc_second_order)
    #print('\n Separated')
    #r = rng(N, size=(M, N, num_resamples))
    #Z = norm.ppf(0.5 + conf_level / 2)

    if True:#not parallel:
        S = create_Si_dict(D, M, num_resamples, keep_resamples, calc_second_order)
        #print('\n Si_dict created')
        for j in range(D):
            S['S1'][j, :] = first_order(A, AB[:, j, :], B, eps)
            #print('\n S1 computed')
            #S1_conf_j = first_order(A[r, :], AB[r, j, :], B[r, :])

            #if keep_resamples:
            #    S['S1_conf_all'][:, j, :] = S1_conf_j

            #S['S1_conf'][j, :] = Z * S1_conf_j.std(ddof=1, axis=0)
            S['ST'][j, :] = total_order(A, AB[:, j, :], B, eps)
            #print('\n ST computed')
            #ST_conf_j = total_order(A[r, :], AB[r, j, :], B[r, :])

            #if keep_resamples:
            #    S['ST_conf_all'][:, j, :] = ST_conf_j

            #S['ST_conf'][j, :] = Z * ST_conf_j.std(ddof=1, axis=0)

        # Second order (+conf.)
        if calc_second_order:
            for j in range(D):
                for k in range(j + 1, D):
                    S['S2'][j, k, :] = second_order(
                        A, AB[:, j, :], AB[:, k, :], BA[:, j, :], B, eps)
                    #print('\n S2 computed')
                    #S['S2_conf'][j, k, :] = Z * second_order(A[r, :], AB[r, j, :],
                    #                                      AB[r, k, :], BA[r, j, :],
                    #                                      B[r, :]).std(ddof=1, axis=0)
    else:
        raise NotImplementedError
        tasks, n_processors = create_task_list(
            D, calc_second_order, n_processors)

        func = partial(sobol_parallel, Z, A, AB, BA, B, r)
        pool = Pool(n_processors)
        S_list = pool.map_async(func, tasks)
        pool.close()
        pool.join()

        S = Si_list_to_dict(S_list.get(), D, num_resamples,
                            keep_resamples, calc_second_order)

    # Add problem context and override conversion method for special case
    S.problem = problem
    #S.to_df = MethodType(to_df, S)

    # Print results to console
    #if print_to_console:
    #    res = S.to_df()
    #    for df in res:
    #        print(df)
    #print('\n On exit')
    return S


def first_order(A, AB, B, eps=1e-20):
    """
    First order estimator following Saltelli et al. 2010 CPC, normalized by
    sample variance
    """
    #return np.mean(B * (AB - A), axis=0) / (eps + np.var(np.r_[A, B], axis=0))
    return safe_division(
        np.mean(B * (AB - A), axis=0), np.var(np.r_[A, B], axis=0), eps=eps, atol=eps*0.1
    )


def total_order(A, AB, B, eps=1e-20):
    """
    Total order estimator following Saltelli et al. 2010 CPC, normalized by
    sample variance
    """
    #return 0.5 * np.mean((A - AB) ** 2, axis=0) / (eps + np.var(np.r_[A, B], axis=0))
    return 0.5*safe_division(
        np.mean((A - AB) ** 2, axis=0), np.var(np.r_[A, B], axis=0), eps=eps, atol=eps*0.1
    )


def second_order(A, ABj, ABk, BAj, B, eps=1e-20):
    """Second order estimator following Saltelli 2002"""
    #Vjk = np.mean(BAj * ABk - A * B, axis=0) / (eps + np.var(np.r_[A, B], axis=0))
    Vjk = safe_division(
        np.mean(BAj * ABk - A * B, axis=0), np.var(np.r_[A, B], axis=0), eps=eps, atol=eps*0.1
    )
    Sj = first_order(A, ABj, B, eps)
    Sk = first_order(A, ABk, B, eps)

    return Vjk - Sj - Sk


def sobol_parallel(Z, A, AB, BA, B, r, tasks):
    sobol_indices = []
    for d, j, k in tasks:
        if d == 'S1':
            s = first_order(A, AB[:, j], B)
        elif d == 'S1_conf':
            s = Z * first_order(A[r], AB[r, j], B[r]).std(ddof=1)
        elif d == 'ST':
            s = total_order(A, AB[:, j], B)
        elif d == 'ST_conf':
            s = Z * total_order(A[r], AB[r, j], B[r]).std(ddof=1)
        elif d == 'S2':
            s = second_order(A, AB[:, j], AB[:, k], BA[:, j], B)
        elif d == 'S2_conf':
            s = Z * second_order(A[r], AB[r, j], AB[r, k],
                                 BA[r, j], B[r]).std(ddof=1)
        sobol_indices.append([d, j, k, s])

    return sobol_indices


def separate_output_values(Y, D, N, calc_second_order):
    M = Y.shape[1]
    AB = np.zeros((N, D, M))
    BA = np.zeros((N, D, M)) if calc_second_order else None
    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0:Y.size:step, :]
    B = Y[(step - 1):Y.size:step, :]
    for j in range(D):
        AB[:, j, :] = Y[(j + 1):Y.size:step, :]
        if calc_second_order:
            BA[:, j] = Y[(j + 1 + D):Y.size:step, :]

    return A, B, AB, BA

def create_Si_dict(D: int, M: int, num_resamples: int, keep_resamples: bool, calc_second_order: bool):
    """initialize empty dict to store sensitivity indices"""
    S = ResultDict((k, np.zeros((D, M)))
                   for k in ('S1', 'S1_conf', 'ST', 'ST_conf'))

    if keep_resamples:
        # Create entries to store intermediate resampling results
        S['S1_conf_all'] = np.empty(0,)#np.zeros((num_resamples, D, M))
        S['ST_conf_all'] = np.empty(0,)#np.zeros((num_resamples, D, M))

    if calc_second_order:
        S['S2'] = np.full((D, D, M), np.nan)
        S['S2_conf'] = np.empty(0,)#np.full((D, D, M), np.nan)

    return S

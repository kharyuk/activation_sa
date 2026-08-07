import numpy as np
import scipy.stats

_corr_types = ('pearson', 'spearman')

def fisher_z(values):#, eps=1e-8):
    return np.arctanh(values)
    #new_values = np.clip(values, -1+eps, 1-eps)
    #return np.arctanh(new_values)

def inv_fisher_z(values):
    return np.tanh(values)

def recover_triu_corr_matrix(a):
    #k*k - k - 2*a.size = 0
    d = (1 + 8*a.size)**0.5
    n = int( (1 + d) / 2 )
    
    i, j = np.triu_indices(n, k=1)
    C = np.empty((n, n), dtype=a.dtype)
    C[i, j] = a
    C[j, i] = a
    np.fill_diagonal(C, 1)
    return C

def custom_spearmanr(a, b=None, axis=0):
    r"""
    Rewritten scipy.stats.spearmanr function.
    """
    assert axis in (0, 1)
    assert 1 <= a.ndim <= 2
    assert (b is not None) or (a.ndim == 2)
    if b is not None:
        if axis == 0:
            a = np.column_stack((a, b))
        else:
            a = np.row_stack((a, b))

    n_vars = a.shape[1 - axis]
    n_obs = a.shape[axis]
    a_ranked = np.apply_along_axis(scipy.stats.rankdata, axis, a)
    rs = np.corrcoef(a_ranked, rowvar=axis)
    is_nan = np.isnan(rs)
    rs[is_nan] = 0
    # rs can have elements equal to 1, so avoid zero division warnings
    if rs.shape == (2, 2):
        rs = rs[0, 1]
        is_nan = is_nan[0, 1]
    return rs, is_nan

    

def compute_unitwise_correlation(
    val_array,
    val_array2=None,
    corr_type='spearman',
    ci_alpha=0.05,
    ci_n_bootstraps=1000,
    ci_random_state=0
):#, n_units=None):
    '''
        val_array has the shape of (n_variables, n_units, n_pixels)
            variables = augmentation variables
            units = output channels / neurons
            
        The function computes unit-wise correlation matrices for every pixel,
        outputs the pixel-wise mean of these matrices.
    '''
    assert corr_type in _corr_types
    C = val_array
    D = None
    n_vars, n_units, n_pixs = C.shape
    cormat_arr = np.zeros([n_pixs, n_vars, n_vars])
    nan_mask = np.zeros([n_pixs, n_vars, n_vars])
    for i in range(n_pixs):
        if val_array2 is not None:
            D = val_array2[:, :, i]
        if corr_type == 'pearson':
            tmp = np.corrcoef(C[:, :, i], D)
            isnan_tmp = np.isnan(tmp)
        elif corr_type == 'spearman':
            #tmp, _ = scipy.stats.spearmanr(
            tmp, isnan_tmp = custom_spearmanr(
                C[:, :, i], D, axis=1#, nan_policy='omit'
            )
        if val_array2 is not None:
            tmp = tmp[:n_vars, n_vars:]
            isnan_tmp = isnan_tmp[:n_vars, n_vars:]
        cormat_arr[i] = tmp
        nan_mask[i] = isnan_tmp
        del tmp, isnan_tmp;
    cormat_arr = fisher_z(cormat_arr)
    mean_cormat = inv_fisher_z(cormat_arr.mean(axis=0))
    if n_pixs == 1:
        del cormat_arr;
        return mean_cormat, None, None, nan_mask
    #std_cormat = cormat_arr.std(axis=0, ddof=1)
    rng = np.random.default_rng(ci_random_state)
    k_bootstrap = len(cormat_arr)
    assert k_bootstrap == n_pixs
    bootstrapped_means = []
    for i_bootstrap in range(ci_n_bootstraps):
        c_ind_bootstrap = rng.choice(k_bootstrap, size=k_bootstrap, replace=True)
        c_mean_bootstrap = cormat_arr[c_ind_bootstrap].mean(axis=0)
        bootstrapped_means.append(c_mean_bootstrap)
    ci_lower = np.quantile(bootstrapped_means, q=0.5*ci_alpha, axis=0)
    ci_upper = np.quantile(bootstrapped_means, q=1-0.5*ci_alpha, axis=0)
    ci_lower = inv_fisher_z(ci_lower)
    ci_upper = inv_fisher_z(ci_upper)
    del cormat_arr, bootstrapped_means;
    return mean_cormat, ci_lower, ci_upper, nan_mask
    #return mean_cormat, std_cormat, nan_mask

def compute_correlation_cov_aug(cov_array, aug_array, corr_type):
    '''
    '''
    assert corr_type in _corr_types
    n_units = len(cov_array)
    pix_shape = None
    if cov_array.ndim > 1:
        pix_shape = cov_array.shape[1:]
    aug_array = aug_array.reshape([n_units, -1])
    cov_array = cov_array.reshape([n_units, -1])
    n_pix = aug_array.shape[1]
    cormat_arr = np.empty((n_pix,))
    nan_mask = np.empty((n_pix,))
    for i in range(n_pix):
        if corr_type == 'pearson':
            cormat_arr[i] = np.corrcoef(
                aug_array[:, i], cov_array[:, i]
            )
        elif corr_type == 'spearman':
            #tmp, _ = scipy.stats.spearmanr(
            tmp, tmp_isnan = custom_spearmanr(
                aug_array[:, i], cov_array[:, i]
            )
            cormat_arr[i] = tmp
            nan_mask[i] = tmp_isnan
    if pix_shape is not None:
        cormat_arr = cormat_arr.reshape(pix_shape)
        nan_mask = nan_mask.reshape(pix_shape)
    return cormat_arr, nan_mask
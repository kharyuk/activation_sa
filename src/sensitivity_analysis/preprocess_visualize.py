import numpy as np

def threshold_pvalues(values, alpha=0.05):
    p = np.ones(values.shape)
    ind = np.where(values >= alpha)
    p[ind] = 0
    return p

def hist_clip_val_fun(a, b, c, d, min_val=-1.88, max_val=1.88):
    return [a] + [np.clip(x, min_val, max_val) for x in [b, c, d]]

def log_scale(values, abs=True):
    if abs:
        return np.log10(1.+np.abs(values))
    return np.log10(1+values)

def compute_coef_var(x, eps=1e-5, log=True):
    # x = (var, mean)
    #y = x[0]/(eps+np.abs(x[1]))
    y = np.sqrt(np.maximum(x[0], 0)) / (np.abs(x[1]) + eps)
    #print(log, y.min(), y.max())
    if log:
        return log_scale(y)
    return y

def log_scale_sum(x, abs=True):
    return log_scale(x.sum(axis=0), abs)

def compute_shpv_coef_var(x, eps=1e-5, log=True):
    #y = x[0].sum(axis=0)/(eps+np.abs(x[1]))
    y = np.sqrt(np.maximum(x[0].sum(axis=0), 0))/(np.abs(x[1]) + eps)
    if log:
        return log_scale(y)
    return y


# Additive property of shpv: summing up all variables related to every single transform/group variable
def extract_shp_values_func(values, group_indices):
    rv = []
    for i in range(len(group_indices)-1):
        ind0, ind1 = group_indices[i:i+2]
        rv.append(np.sum(values[ind0:ind1], axis=0))
    rv = np.array(rv)
    rv = np.clip(rv, 0., None)
    return rv

def hist_shp_values_func(values, group_indices):
    rv = []
    for i in range(len(group_indices)-1):
        ind0, ind1 = group_indices[i:i+2]
        rv.append(np.sum(values[ind0:ind1], axis=0))
        #rv.append(np.sum(np.clip(values[ind0:ind1], 0., None), axis=0))
    return np.clip(np.array(rv), 0., None)
    #return np.array(rv)
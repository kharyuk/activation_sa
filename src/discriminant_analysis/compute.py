import functools

import sklearn.discriminant_analysis
import sklearn.metrics

import scipy.stats
import numpy as np

import preparation.single_unit


def compute_mean_valid_confusion_matrices(
    sensitivity_values_dict,
    network_modules,
    augmentation_set_numbers_list,
    values_names,
    n_conv_modules,
    repeated_kfold_n_splits=5,
    repeated_kfold_n_repeats=100,
    random_state=0,
    extract_auxilliary_names=True,
):

    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )


    #val_names = ['shpv', 'si', 'siT']
    #n_conv_modules = 5

    #random_state = 24678943
    rng = np.random.default_rng(random_state)

    repeated_kfold_random_states = rng.integers(
        0, 100000, size=(n_conv_modules, len(augmentation_set_numbers_list))
    )
    
    # val_name -> module_name -> aug_set_num
    result_cms = {}
    result_acc = {}
    
    for i_cval, cur_vals_name in enumerate(values_names):
        result_cms[cur_vals_name] = result_cms.get(cur_vals_name, {})
        result_acc[cur_vals_name] = result_acc.get(cur_vals_name, {})
        for i_mn, module_name in enumerate(network_modules):
            if i_mn == n_conv_modules:
                break
            result_cms[cur_vals_name][module_name] = result_cms[cur_vals_name].get(module_name, {})
            result_acc[cur_vals_name][module_name] = result_acc[cur_vals_name].get(module_name, {})
            i_var = 0
            for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
                result_cms[cur_vals_name][module_name][augmentation_set_number] = (
                    result_cms[cur_vals_name][module_name].get(augmentation_set_number, {})
                )
                result_acc[cur_vals_name][module_name][augmentation_set_number] = (
                    result_acc[cur_vals_name][module_name].get(augmentation_set_number, {})
                )
                n_vars = len(augmentation_names_dict[augmentation_set_number])
                repeated_kfold_cv = sklearn.model_selection.RepeatedKFold(
                    n_splits=repeated_kfold_n_splits,
                    n_repeats=repeated_kfold_n_repeats,
                    random_state=repeated_kfold_random_states[i_mn, i_aug],
                )
                
                X = sensitivity_values_dict[cur_vals_name][module_name][i_var:i_var+n_vars]
                n_vars, n_units = X.shape[:2]
                X = X.reshape((n_vars, n_units, -1))
                n_spat = X.shape[-1]
                y = np.arange(n_vars)[:, None]
                y = np.tile(y, (1, n_units))
                X = X.reshape((n_vars*n_units, n_spat))
                y = y.reshape((n_vars*n_units, ))
                
                conf_mat_train_list, conf_mat_valid_list = [], []
                acc_train, acc_valid = [], []
                for i_cv, (train_ind, valid_ind) in enumerate(repeated_kfold_cv.split(X)):
                    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
                        solver='svd',
                        shrinkage=None,
                        priors=None,
                        n_components=2,
                        store_covariance=True,
                        tol=0.0001,
                        covariance_estimator=None,
                    )
                    lda.fit(X[train_ind], y[train_ind])
                    y_pred_train = lda.predict(X[train_ind])
                    y_pred_valid = lda.predict(X[valid_ind])
                    acc_train.append((y[train_ind] == y_pred_train).sum() / len(train_ind))
                    acc_valid.append((y[valid_ind] == y_pred_valid).sum() / len(valid_ind))

                    conf_mat_train = sklearn.metrics.confusion_matrix(y[train_ind], y_pred_train)#, normalize='true')
                    conf_mat_valid = sklearn.metrics.confusion_matrix(y[valid_ind], y_pred_valid)#, normalize='true')

                    conf_mat_train_list.append(conf_mat_train)
                    conf_mat_valid_list.append(conf_mat_valid)
                    
                    del conf_mat_train, conf_mat_valid, y_pred_train, y_pred_valid, lda;
                    
                conf_mat_train = np.sum(conf_mat_train_list, axis=0)
                conf_mat_valid = np.sum(conf_mat_valid_list, axis=0)
                # sum(axis=1) means that we get number of all true samples per variable
                conf_mat_train = conf_mat_train / np.sum(conf_mat_train, axis=1, keepdims=True)
                conf_mat_valid = conf_mat_valid / np.sum(conf_mat_valid, axis=1, keepdims=True)
                
                result_cms[cur_vals_name][module_name][augmentation_set_number]['train'] = conf_mat_train
                result_cms[cur_vals_name][module_name][augmentation_set_number]['valid'] = conf_mat_valid
                
                result_acc[cur_vals_name][module_name][augmentation_set_number]['train'] = np.mean(acc_train)
                result_acc[cur_vals_name][module_name][augmentation_set_number]['valid'] = np.mean(acc_valid)
                
                del repeated_kfold_cv, conf_mat_train_list, conf_mat_valid_list, conf_mat_train, conf_mat_valid;
                
                i_var += n_vars
    return result_cms, result_acc


def project_2d_lda(
    sensitivity_values_dict,
    network_modules,
    augmentation_set_numbers_list,
    values_names,
    n_conv_modules,
    extract_auxilliary_names=True,
):
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    
    projected_sensitivity_values = {}
    
    for i_cval, cur_vals_name in enumerate(values_names):
        projected_sensitivity_values[cur_vals_name] = projected_sensitivity_values.get(cur_vals_name, {})
        for i_mn, module_name in enumerate(network_modules):
            if i_mn == n_conv_modules:
                break
            projected_sensitivity_values[cur_vals_name][module_name] = (
                projected_sensitivity_values[cur_vals_name].get(module_name, {})
            )
            i_var = 0
            for i_aug, augmentation_set_number in enumerate(augmentation_set_numbers_list):
                n_vars = len(augmentation_names_dict[augmentation_set_number])
                
                X = sensitivity_values_dict[cur_vals_name][module_name][i_var:i_var+n_vars]
                n_vars, n_units = X.shape[:2]
                X = X.reshape((n_vars, n_units, -1))
                n_spat = X.shape[-1]
                y = np.arange(n_vars)[:, None]
                y = np.tile(y, (1, n_units))
                X = X.reshape((n_vars*n_units, n_spat))
                y = y.reshape((n_vars*n_units, ))
                
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
                    solver='svd',
                    shrinkage=None,
                    priors=None,
                    n_components=2,
                    store_covariance=True,
                    tol=0.0001,
                    covariance_estimator=None,
                )
                X_tr_lda = lda.fit(X, y).transform(X)
                projected_sensitivity_values[cur_vals_name][module_name][augmentation_set_number] = X_tr_lda
                
                del X_tr_lda;
                
                i_var += n_vars
        
    return projected_sensitivity_values




def remap_dist(
    Z, y, distribution, dist_parameters
):
    # project data to inner copula space
    X = Z.copy()
    classes = set(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape
    # to estiate the common within-class covariance matrix
    M = np.empty((n_classes, n_features))
    W = np.zeros((n_features, n_features))
    class_priors = np.empty(n_classes)
    for i, cl in enumerate(classes):
        cX = X[y==cl]
        c_mean = cX.mean(axis=0)
        #M[i] = c_mean
        for j in range(n_features):
            c_dist = distribution(*dist_parameters[i][j])
            
            cX[:, j] = c_dist.cdf(cX[:, j])
            cX[:, j] = scipy.stats.norm.ppf(cX[:, j])
        X[y==cl] = cX
    return X
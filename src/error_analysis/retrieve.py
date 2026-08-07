import numpy as np
import preparation.visualize

def load_values(
    values_fnms_dict,
    activations_dirname,
    network_modules_list,
    values_key,
    values_name,
    augmentation_set_numbers_list,
    values_func=None,
    post_values_func=None
):
    n_modules = len(network_modules_list)
    vals_dict = dict()
    for i_mn, module_name in enumerate(network_modules_list):
        values = {}
        for augmentation_set_number in augmentation_set_numbers_list:
            current_values = preparation.visualize.get_conv2d_unit_values(
                values_fnms_dict,
                activations_dirname,
                module_name,
                values_key=values_key,
                values_name=values_name,
                augmentation_set_number=augmentation_set_number,
                dataset_part=None,
                slice_num=None,
                values_func=values_func,
                shpv_normalize=False,
            )
            if post_values_func is not None:
                current_values = post_values_func(current_values)
            values[augmentation_set_number] = current_values
        vals_dict[module_name] = values
    return vals_dict

def retrieve_sobol_outs(
    si_values,
    si_variances,
    x_si,
    x_sT,
    network_modules,
    augmentation_set_numbers_list,
    #eps=1e-5
):
    y_si = dict()
    cnt_si = dict()
    y_sT = dict()
    cnt_sT = dict()
    for module_name in network_modules:
        y_si[module_name] = dict()
        y_sT[module_name] = dict()
        cnt_si[module_name] = dict()
        cnt_sT[module_name] = dict()
        for aug_set in augmentation_set_numbers_list:
            c_values = si_values[module_name][aug_set]
            c_variances = si_variances[module_name][aug_set][0]
            c_values = np.reshape(c_values, (2, -1) + c_values.shape[1:])
            #print(c_values.shape, c_variances.shape)
            N = c_values[0].size
            cy = []
            c_cnt = []
            for cx in x_si:
                ind_neg_si = np.where(c_values[0] < cx)
                N_neg = len(ind_neg_si[0])
                c_cnt.append(N_neg / N)
                if N_neg == 0:
                    cy.append(0.)
                    continue
                variances_neg_si = c_variances[ind_neg_si[1:]]
                #cy.append( np.max(variances_neg_si) )
                cy.append( np.max(variances_neg_si) / (np.max(c_variances) - np.min(c_variances) ) )
            y_si[module_name][aug_set] = np.array(cy)
            cnt_si[module_name][aug_set] = np.array(c_cnt)
            N = c_values[1].size
            cy = []
            c_cnt = []
            for cx in x_sT:
                ind_high_sT = np.where(c_values[1] > cx)
                N_high = len(ind_high_sT[0])
                c_cnt.append(N_high / N)
                if N_high == 0:
                    cy.append(0.)
                    continue
                variances_high_sT = c_variances[ind_high_sT[1:]]
                #cy.append( np.max(variances_high_sT) )
                cy.append( np.max(variances_high_sT) / (np.max(c_variances) - np.min(c_variances) ) )
            y_sT[module_name][aug_set] = np.array(cy)
            cnt_sT[module_name][aug_set] = np.array(c_cnt)
    return y_si, cnt_si, y_sT, cnt_sT

def retrieve_shapley_outs(
    shpv_values,
    shpv_variances,
    x_shpv,
    network_modules,
    augmentation_set_numbers_list,
    #eps=1e-5
):
    y_shpv = dict()
    cnt_shpv = dict()

    for module_name in network_modules:
        y_shpv[module_name] = dict()
        cnt_shpv[module_name] = dict()
        #uvf_shpv[module_name] = dict()
        for aug_set in augmentation_set_numbers_list:
            c_values = shpv_values[module_name][aug_set]
            c_variances = shpv_variances[module_name][aug_set][0]
            #c_variances_est = shpv_variances_est[module_name][aug_set]
            c_values = np.reshape(c_values, (-1, ) + c_values.shape[1:])
            #print(c_values.shape, c_variances.shape)
            N = c_values.size
            cy = []
            c_cnt = []
            #uvf = []
            for cx in x_shpv:
                ind_neg_shpv = np.where(c_values < cx)
                N_neg = len(ind_neg_shpv[0])
                c_cnt.append(N_neg / N)
                if N_neg == 0:
                    cy.append(0.)
                    #uvf.append(0.)
                    continue
                sum_values_neg_shpv = -c_values[ind_neg_shpv].sum(axis=0)
                variances_neg_shpv = c_variances[ind_neg_shpv[1:]]
                #variance_ratio = c_values[ind_neg_shpv].min() / c_values[ind_neg_shpv].max()
                #variance_ratio = (c_variances / (1e-5 + variances_neg_shpv)).max()
                
                #cy.append( np.max(variances_neg_shpv) )
                cy.append( np.max(variances_neg_shpv) / (np.max(c_variances) - np.min(c_variances)) )
                #cy.append( variance_ratio )
                #variance_ratio = (1. - (c_variances_est[ind_neg_shpv[1:]] / (eps + variances_neg_shpv))).max()
                #variance_ratio = (c_variances_est[ind_neg_shpv[1:]] / (1e-5 + variances_neg_shpv))).max()
                #variance_ratio = (sum_values_neg_shpv / (1e-5 + variances_neg_shpv)).max()
                #uvf.append(variance_ratio)
            y_shpv[module_name][aug_set] = np.array(cy)
            cnt_shpv[module_name][aug_set] = np.array(c_cnt)
            #uvf_shpv[module_name][aug_set] = np.array(uvf)
    return y_shpv, cnt_shpv
import logging
import functools
import itertools

import h5py
import numpy as np
import torch
import pandas as pd

import torch_utils
import torch_learning

import sensitivity_analysis.utils
import sensitivity_analysis.contrast

import preparation.single_unit
import preparation.visualize

def get_all_values(
    values_fnms_dict,
    activations_dirname,
    network_modules_list,
    values_name,
    augmentation_set_numbers_list,
    shpv_group_indices_dict,
    extract_auxilliary_names=False,
):

    n_modules = len(network_modules_list)

    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    augmentation_names = functools.reduce(
        lambda x, y: x+y, augmentation_names_dict.values()
    )
    n_aug = len(augmentation_names)
    
    rv = {}

    for i_mn, module_name in enumerate(network_modules_list):
        value_key = preparation.visualize._values_keys_dict[values_name]
        dataset_part = None
        if 'rbscc' in values_name:
            _, dataset_part = row_name.split(' ')
            dataset_part = dataset_part.replace('(', '').replace(')', '')
        c_values_name = preparation.visualize._values_names_dict[value_key]
        values = []
        for i_col in range(n_aug):
            aug_name = augmentation_names[i_col]
            slice_num = i_col
            for augmentation_set_number in augmentation_set_numbers_list:
                L = len(augmentation_names_dict[augmentation_set_number])
                if slice_num >= L:
                    slice_num -= L
                else:
                    break
            current_values = preparation.visualize.get_conv2d_unit_values(
                values_fnms_dict,
                activations_dirname,
                module_name,
                values_key=value_key,
                values_name=c_values_name,
                augmentation_set_number=augmentation_set_number,
                dataset_part=dataset_part,
                slice_num=slice_num,
                values_func=None if values_name not in preparation.visualize._values_funcs else lambda x: (
                    preparation.visualize._values_funcs[values_name](
                        x,
                        shpv_group_indices_dict[augmentation_set_number]
                    )
                ),
                shpv_normalize=True,
            )
            values.append(current_values)
        values = np.array(values)
        rv[module_name] = values

    return rv

def get_no_augaux_values(
    values_fnms_dict,
    activations_dirname,
    network_modules_list,
    augmentation_set_numbers_list,
    value_key,
    values_name='mean',
):

    n_modules = len(network_modules_list)
    n_augsets = len(augmentation_set_numbers_list)
    rv = {}
    for i_mn, module_name in enumerate(network_modules_list):
        values = []
        for augmentation_set_number in augmentation_set_numbers_list:
            current_values = preparation.visualize.get_conv2d_unit_values(
                values_fnms_dict,
                activations_dirname,
                module_name,
                values_key=value_key,
                values_name=values_name,
                augmentation_set_number=augmentation_set_number,
                dataset_part=None,
                slice_num=None,
            )
            values.append(current_values)
        values = np.array(values)
        rv[module_name] = values
    return rv


def _create_data_record(group, values, val_name='values'):
    group.create_dataset(
        name=val_name,
        data=values,                
        compression="gzip",
        chunks=True,
        maxshape=(None, ) + values.shape[1:]
    )

def _add_data(group, new_values, val_name='values'):
    len_a = group[val_name].shape[0]
    len_b = len(new_values)
    group[val_name].resize(len_a+len_b, axis=0)
    group[val_name][-len_b:] = new_values
    
def save_predictions_to_hdf5(
    hdf5_path,
    y,
    group_name,
    top_n=5,
    alpha=None,
    percentile=None,
    aug_name=None,
    invert=None,
    subgroup_name=None,
):
    y = y.detach().cpu().numpy().astype('float32')
    ndim_y = y.ndim
    
    y_true_flag = (alpha is None) and (percentile is None) and (aug_name is None) and (ndim_y < 3)
    y_quantile_mask_flag = (alpha is not None) and (percentile is not None) and (aug_name is not None) and (invert is None)
    y_full_mask_flag = (alpha is None) and (percentile is None) and (aug_name is not None) and (invert is not None)
    y_no_mask_flag = (alpha is None) and (percentile is None) and (aug_name is not None)
    assert y_true_flag or y_quantile_mask_flag or y_full_mask_flag or y_no_mask_flag
    
    len_b = len(y)
    if not y_true_flag:
        ind = np.argsort(y, axis=-1).T
        ind = ind[::-1][:top_n].T
        y = np.sort(y, axis=-1).T
        y = y[::-1][:top_n].T
        ind = np.reshape(ind, (len_b, -1))
        
    shape = y.shape[1:]
    y = np.reshape(y, (len_b, -1))
    
    with h5py.File(hdf5_path, 'a') as fd:
        if group_name in fd:
            current_group = fd[group_name]
            if y_true_flag:
                _add_data(current_group, y.astype('int16'), val_name='labels')
        else:
            current_group = fd.create_group(group_name)
            if y_true_flag:
                _create_data_record(current_group, y.astype('int16'), val_name='labels')
                current_group.attrs['shape'] = shape
                
        if not y_true_flag:            
            if subgroup_name is None:
                subgroup_name = ''
            if (alpha is not None):
                subgroup_name += f'alpha={alpha:.2f}'
            if (percentile is not None):
                subgroup_name += f'_perc={percentile:.2f}'
            if (aug_name is not None):
                subgroup_name += f'_iaug={aug_name}'
            if (invert is not None):
                subgroup_name += f'_invert={int(invert)}'
            if subgroup_name.startswith('_'):
                subgroup_name = subgroup_name[1:]
            if subgroup_name in current_group:
                current_subgroup = current_group[subgroup_name]
                _add_data(current_subgroup, ind, val_name='predictions')
                _add_data(current_subgroup, y, val_name='probas')
            else:
                current_subgroup = current_group.create_group(subgroup_name)
                _create_data_record(current_subgroup, ind, val_name='predictions')
                _create_data_record(current_subgroup, y, val_name='probas')
                current_subgroup.attrs['shape'] = shape
    
    
def modify_activations(par_list, sensitivity_values_dict, name, mod, inputs, outputs):
    #print('modify within: ', par_list)
    alpha, perc, aug_ind, invert = par_list
    y_alpha_perc_aug_flag = (alpha is not None) and (perc is not None) and (aug_ind is not None)
    y_aug_flag = (alpha is None) and (perc is None) and (aug_ind is not None) and (invert is not None)
    if not (y_alpha_perc_aug_flag or y_aug_flag):
        #print('no_modifications')
        return outputs
    ndim = len(outputs.shape)-1
    n_batch = outputs.shape[0]
    mask = sensitivity_values_dict[name][aug_ind].repeat((n_batch,)+(1,)*ndim).to(
        dtype=outputs.dtype, device=outputs.device
    )
    if y_aug_flag:
        if invert:
            return outputs*(1-mask)
        return outputs*mask
    if y_alpha_perc_aug_flag:
        qv = torch.quantile(sensitivity_values_dict[name][aug_ind], perc)
        return torch.where(
            mask < qv,
            outputs,
            alpha*outputs,
        )
    
def compute_predictions_wrt_single_augs(
    sampler,
    model,
    module_names,
    transform_functions_list,
    inference_params,
    dataset,
    batch_size,
    Nsamples,
    Ninner_samples,
    N_aux,
    augmentation_set_numbers_list,
    sensitivity_values_dict,
    alphas,
    percentiles,
    save_path,
    top_n=5,
    #save_basename='modified_activations_pred',
    log_path='modified_activations_pred.log',
    pre_processing_functions=None,
    post_processing_functions=None,
    verbose=True,
    device='cpu',
    trace_sampled_params_path=None,
):
    '''
    class variable is always the last one; group name and variable name for
    class are supposed to be identical
    '''
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=True,
    )
    
    if log_path is not None:
        torch_learning.initialize_logging(log_path)
    
    if trace_sampled_params_path is not None:
        traced_params = {}
    
    alpha, perc, i_aug2, invert = None, None, None, None # dirty hack (global variable)
    par_list = [alpha, perc, i_aug2, invert]
    modify_activations_function = lambda name, mod, inputs, outputs: modify_activations(
        par_list, sensitivity_values_dict, name, mod, inputs, outputs
    )
    handlers = torch_utils.register_forward_hook_by_name(
        model, modify_activations_function, module_names
    )
    Naugs = sampler.num_augs
    print(Naugs, sampler.augs)
    #N_aux = 3 # hardcoded
    #caps = CustomAugParamSampler(problem, augpar_sampler_seeds)
    
    no_aug_key = 'original'
    y_true_key = 'true_labels'
    
    model.eval();
    model = model.to(device)
    partX, y_true = [], []
    y_pred_quantile_mask = {}
    y_pred_full_mask = {}
    y_pred_no_mask = {}
    n_samplesX, total_samplesX = 0, 0
    info_msg = 'Started.'
    if log_path is not None:
        logging.info(info_msg)
    N_samples_left = Nsamples
    with torch.no_grad():
        #for i_sample in range(Nsamples):
        while N_samples_left > 0:
            cur_batch_size = min(batch_size, N_samples_left)
            sampleV, indX = sampler.sample(Nouter=cur_batch_size, Ninner=Ninner_samples)
            #indX = indX[0]
            #sampleV = sampleV[0]
            if trace_sampled_params_path is not None:
                for i in range(cur_batch_size):
                    traced_params = sensitivity_analysis.utils.custom_merge_dicts(traced_params, sampleV[i])
                    traced_params = sensitivity_analysis.utils.custom_merge_dicts(traced_params, {'class': indX[i]})
            
            sampleX = []
            for ind in indX:
                tmp_X, tmp_Y = dataset[ind]
                if pre_processing_functions is not None:
                    tmp_X = pre_processing_functions(tmp_X)
                sampleX.append(tmp_X)
                y_true.append(tmp_Y)
                #del tmp_X, tmp_Y;
            sampleX = torch.stack(sampleX, dim=0)
            #partX = []
            if post_processing_functions is not None:
                partX = post_processing_functions(sampleX.clone())
                #for i in range(cur_batch_size):
                #    partX.append(post_processing_functions(sampleX[i]))
            #n_samplesX += 1
            N_samples_left -= cur_batch_size
            
            # non-augmented input: non-masked predictions
            #partX = torch.stack(partX, dim=0)
            #print('2222', partX.shape)
            for l in range(len(par_list)):
                par_list[l] = None
            ##print('non-augmented input: non-masked predictions', par_list)
            predY = model(partX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
            #y_pred_no_mask[no_aug_key] = y_pred_no_mask.get(no_aug_key, []) + [predY]
            y_pred_no_mask[no_aug_key] = predY
            #print(y_true, [x.argmax() for x in predY.detach().numpy()], partX.min(), partX.max())
            #jkkhhj
            del predY;
            #####################################
            
            for i_aug in range(Naugs): # select augmentation for batch
                aug_num = i_aug
                aug_offset = 0
                for augmentation_set_number in augmentation_set_numbers_list:
                    L = len(augmentation_names_dict[augmentation_set_number]) - N_aux
                    if aug_num >= L:
                        aug_num -= L
                        aug_offset += L+N_aux
                    else:
                        break
                
                aug_name = sampler.augs[i_aug]#problem['aug_names'][i_aug]
                y_pred_quantile_mask[aug_name] = y_pred_quantile_mask.get(aug_name, {})
                y_pred_full_mask[aug_name] = y_pred_full_mask.get(aug_name, {})
                
                copy_sampleX = []
                #print('0000', sampleX.shape)
                for i in range(cur_batch_size):
                    aug_params = sampleV[i][aug_name]
                    copy_sampleX.append(
                        sensitivity_analysis.contrast.sample_transform(
                            #sampleX[i].clone(), transform_functions_list[i_aug], aug_params
                            sampleX[i], transform_functions_list[i_aug], aug_params
                        )
                    )
                #print('0011', [x.shape for x in copy_sampleX])
                if copy_sampleX[0].dim() == 3:
                    copy_sampleX = torch.stack(copy_sampleX, dim=0) #.unsqueeze(0)
                else:
                    copy_sampleX = torch.cat(copy_sampleX, dim=0)                
                #print('11111', copy_sampleX.shape)
                if post_processing_functions is not None:
                    copy_sampleX = post_processing_functions(copy_sampleX)
                # augmented input: predict with non-modified activations
                for l in range(len(par_list)):
                    par_list[l] = None
                ##print('augmented input: predict with non-modified activations', par_list)
                predY = model(copy_sampleX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                #y_pred_no_mask[aug_name] = y_pred_no_mask.get(aug_name, []) + [predY]
                y_pred_no_mask[aug_name] = predY
                del predY;
                #####################################
                n_cur_aug = len(augmentation_names_dict[augmentation_set_number])
                for i_aug2 in range(n_cur_aug):
                    par_list[0] = par_list[1] = None
                    par_list[2] = aug_offset + i_aug2
                    aug_name2 = augmentation_names_dict[augmentation_set_number][i_aug2]
                    if i_aug2 >= n_cur_aug-N_aux:
                        aug_name2 = f'{aug_name2}-{augmentation_set_number}'
                    # augmented input: predict with full-sized masked activations
                    for invert in [False, True]:
                        par_list[3] = invert
                        ##print('augmented input: predict with full-sized masked activations', par_list)
                        predY = model(copy_sampleX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                        loc_key = (aug_name2, invert)
                        #y_pred_full_mask[aug_name][loc_key] = (
                        #    y_pred_full_mask[aug_name].get(loc_key, []) + [predY]
                        #)
                        y_pred_full_mask[aug_name][loc_key] = predY
                        del predY;
                        
                    #################################
                    # augmented input: predict with quantile-selected masked activations
                    par_list[3] = None
                    for i_alpha, alpha in enumerate(alphas):
                        par_list[0] = alpha
                        for i_perc, perc in enumerate(percentiles):
                            par_list[1] = perc
                            ##print('augmented input: predict with quantile-selected masked activations', par_list)
                            predY = model(copy_sampleX.to(device), **inference_params).to(device='cpu')##.detach().numpy()
                            loc_key = (alpha, perc, aug_name2)
                            #y_pred_quantile_mask[aug_name][loc_key] = (
                            #    y_pred_quantile_mask[aug_name].get(loc_key, []) + [predY]
                            #)
                            y_pred_quantile_mask[aug_name][loc_key] = predY
                            del predY;
                    ###############################
                del copy_sampleX;
                
            del sampleX;
                    
            total_samplesX += cur_batch_size
            info_msg = f'sampled={total_samplesX}/{Nsamples}'
            if verbose:
                print(info_msg, end='\r')
            if log_path is not None:
                logging.info(info_msg)
                
            
            aug_offset = 0
            y_pred_full_mask[no_aug_key] = y_pred_full_mask.get(no_aug_key, {})
            y_pred_quantile_mask[no_aug_key] = y_pred_quantile_mask.get(no_aug_key, {})
            for augmentation_set_number in augmentation_set_numbers_list:
                n_cur_aug = len(augmentation_names_dict[augmentation_set_number])
                for i_aug2 in range(n_cur_aug):
                    par_list[0] = par_list[1] = None
                    par_list[2] = aug_offset + i_aug2
                    aug_name2 = augmentation_names_dict[augmentation_set_number][i_aug2]
                    if i_aug2 >= n_cur_aug-N_aux:
                        aug_name2 = f'{aug_name2}-{augmentation_set_number}'
                    # non-augmented input: predict with full-sized masked activations
                    for invert in [False, True]:
                        par_list[3] = invert
                        ##print('non-augmented input: predict with full-sized masked activations', par_list)
                        predY = model(partX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                        loc_key = (aug_name2, invert)
                        #y_pred_full_mask[no_aug_key][loc_key] = (
                        #    y_pred_full_mask[no_aug_key].get(loc_key, []) + [predY]
                        #)
                        y_pred_full_mask[no_aug_key][loc_key] = predY
                        del predY;
                    #################################
                    # augmented input: predict with quantile-selected masked activations
                    par_list[3] = None
                    for i_alpha, alpha in enumerate(alphas):
                        par_list[0] = alpha
                        for i_perc, perc in enumerate(percentiles):
                            par_list[1] = perc
                            ##print('augmented input: predict with quantile-selected masked activations', par_list)
                            predY = model(partX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                            loc_key = (alpha, perc, aug_name2)
                            #y_pred_quantile_mask[no_aug_key][loc_key] = (
                            #    y_pred_quantile_mask[no_aug_key].get(loc_key, []) + [predY]
                            #)
                            y_pred_quantile_mask[no_aug_key][loc_key] = predY
                            del predY;
                    ###############################
                aug_offset += n_cur_aug
            #################################
            
            # y_true
            y_true = torch.from_numpy(np.array(y_true))
            save_predictions_to_hdf5(
                save_path,
                y_true,
                #torch.cat([y_true], dim=0),
                group_name=y_true_key,
                top_n=top_n,
                alpha=None,
                percentile=None,
                aug_name=None,
                invert=None,
            )
            ###################################
            for aug_key in y_pred_quantile_mask:
                for alpha_perc_aug in y_pred_quantile_mask[aug_key]:
                    alpha, perc, aug_name2 = alpha_perc_aug
                    #print(torch.stack(y_pred_quantile_mask[aug_key][alpha_perc_aug], dim=0).shape)
                    #print(y_pred_quantile_mask[aug_key][alpha_perc_aug][0].shape)
                    #print(len(y_pred_quantile_mask[aug_key][alpha_perc_aug]))
                    cur_y = y_pred_quantile_mask[aug_key][alpha_perc_aug]
                    #cur_y = cur_y.reshape(
                    #    (cur_batch_size, -1)+cur_y.shape[1:]
                    #)
                    #if aug_key == no_aug_key:
                    #    cur_y = torch.cat(cur_y, dim=0)
                    #else:
                    #    cur_y = torch.stack(cur_y, dim=0)
                    save_predictions_to_hdf5(
                        save_path,
                        cur_y,
                        group_name=aug_key,
                        top_n=top_n,
                        alpha=alpha,
                        percentile=perc,
                        aug_name=aug_name2,
                        invert=None,
                    )
                    del cur_y;
            for aug_key in y_pred_full_mask:
                for aug_invert in y_pred_full_mask[aug_key]:
                    aug_name2, invert = aug_invert
                    cur_y = y_pred_full_mask[aug_key][aug_invert]
                    #cur_y = cur_y.reshape(
                    #    (cur_batch_size, -1)+cur_y.shape[1:]
                    #)
                    #if aug_key == no_aug_key:
                    #    cur_y = torch.cat(cur_y, dim=0)
                    #else:
                    #    cur_y = torch.stack(cur_y, dim=0)
                    save_predictions_to_hdf5(
                        save_path,
                        cur_y,
                        group_name=aug_key,
                        top_n=top_n,
                        alpha=None,
                        percentile=None,
                        aug_name=aug_name2,
                        invert=invert,
                    )
                    del cur_y;
            for aug_key in y_pred_no_mask:
                cur_y = y_pred_no_mask[aug_key]
                #cur_y = cur_y.reshape(
                #    (cur_batch_size, -1)+cur_y.shape[1:]
                #)
                #if aug_key == no_aug_key:
                #    cur_y = torch.cat(cur_y, dim=0)
                #else:
                #    cur_y = torch.stack(cur_y, dim=0)
                save_predictions_to_hdf5(
                    save_path,
                    cur_y,
                    group_name=aug_key,
                    top_n=top_n,
                    alpha=None,
                    percentile=None,
                    aug_name=no_aug_key,
                    invert=None,
                )
                del cur_y;
            del partX, y_pred_quantile_mask, y_true, y_pred_full_mask, y_pred_no_mask;
            partX, y_true = [], []
            y_pred_quantile_mask = {}
            y_pred_full_mask = {}
            y_pred_no_mask = {}
            #n_samplesX = 0
    torch_utils.remove_all_hooks_by_dict(handlers)
    info_msg = f'Finished.'
    if verbose:
        print(f'\n{info_msg}')
    if log_path is not None:
        logging.info(info_msg)
    if trace_sampled_params_path is not None:
        np.savez_compressed(trace_sampled_params_path, **traced_params)
    return save_path



    
    
'''
    
def compute_predictions_wrt_single_augs_old(
    sampler,
    model,
    module_names,
    transform_functions_list,
    inference_params,
    dataset,
    batch_size,
    Nsamples,
    Ninner_samples,
    augmentation_set_numbers_list,
    sensitivity_values_dict,
    alphas,
    percentiles,
    save_path,
    top_n=5,
    #save_basename='modified_activations_pred',
    log_path='modified_activations_pred.log',
    pre_processing_functions=None,
    post_processing_functions=None,
    verbose=True,
    device='cpu',
    trace_sampled_params_path=None,
):
    ''
    class variable is always the last one; group name and variable name for
    class are supposed to be identical
    ''
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=True,
    )
    
    if log_path is not None:
        torch_learning.initialize_logging(log_path)
    
    if trace_sampled_params_path is not None:
        traced_params = {}
    
    alpha, perc, i_aug2, invert = None, None, None, None # dirty hack (global variable)
    par_list = [alpha, perc, i_aug2, invert]
    modify_activations_function = lambda name, mod, inputs, outputs: modify_activations(
        par_list, sensitivity_values_dict, name, mod, inputs, outputs
    )
    handlers = torch_utils.register_forward_hook_by_name(
        model, modify_activations_function, module_names
    )
    Naugs = sampler.num_augs
    N_augs2 = len(functools.reduce(lambda x, y: x+y, augmentation_names_dict.values()))
    N_aux = 3 # hardcoded
    #caps = CustomAugParamSampler(problem, augpar_sampler_seeds)
    
    no_aug_key = 'original'
    y_true_key = 'true_labels'
    
    model.eval();
    model = model.to(device)
    partX, y_true = [], []
    y_pred_quantile_mask = {}
    y_pred_full_mask = {}
    y_pred_no_mask = {}
    n_samplesX, total_samplesX = 0, 0
    info_msg = 'Started.'
    if log_path is not None:
        logging.info(info_msg)
    with torch.no_grad():
        for i_sample in range(Nsamples):
            sampleV, indX = sampler.sample(Nouter=1, Ninner=Ninner_samples)
            indX = indX[0]
            sampleV = sampleV[0]
            if trace_sampled_params_path is not None:
                traced_params = sensitivity_analysis.utils.custom_merge_dicts(traced_params, sampleV)
                traced_params = sensitivity_analysis.utils.custom_merge_dicts(traced_params, {'class': indX})
            
            sampleX, sampleY = dataset[indX]
            y_true.append(sampleY)
            if pre_processing_functions is not None:
                sampleX = pre_processing_functions(sampleX)
            #print(sampleX.shape)
            if post_processing_functions is not None:
                partX.append(post_processing_functions(sampleX.clone()))
            n_samplesX += 1
            
            for i_aug in range(Naugs): # select augmentation for batch
                aug_num = i_aug
                aug_offset = 0
                for augmentation_set_number in augmentation_set_numbers_list:
                    L = len(augmentation_names_dict[augmentation_set_number]) - N_aux
                    if aug_num >= L:
                        aug_num -= L
                        aug_offset += L+N_aux
                    else:
                        break
                
                aug_name = sampler.augs[i_aug]#problem['aug_names'][i_aug]
                y_pred_quantile_mask[aug_name] = y_pred_quantile_mask.get(aug_name, {})
                y_pred_full_mask[aug_name] = y_pred_full_mask.get(aug_name, {})
                aug_params = sampleV[aug_name]
                copy_sampleX = sensitivity_analysis.contrast.sample_transform(
                    sampleX.clone(), transform_functions_list[i_aug], aug_params
                )
                if post_processing_functions is not None:
                    copy_sampleX = post_processing_functions(copy_sampleX)
                if copy_sampleX.dim() == 3:
                    copy_sampleX = copy_sampleX.unsqueeze(0)
                # augmented input: predict with non-modified activations
                predY = model(copy_sampleX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                y_pred_no_mask[aug_name] = y_pred_no_mask.get(aug_name, []) + [predY]
                del predY;
                #####################################
                n_cur_aug = len(augmentation_names_dict[augmentation_set_number])
                for i_aug2 in range(n_cur_aug):
                    par_list[2] = aug_offset + i_aug2
                    aug_name2 = augmentation_names_dict[augmentation_set_number][i_aug2]
                    if i_aug2 >= n_cur_aug-N_aux:
                        aug_name2 = f'{aug_name2}-{augmentation_set_number}'
                    # augmented input: predict with full-sized masked activations
                    for invert in [False, True]:
                        par_list[3] = invert
                        predY = model(copy_sampleX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                        loc_key = (aug_name2, invert)
                        y_pred_full_mask[aug_name][loc_key] = (
                            y_pred_full_mask[aug_name].get(loc_key, []) + [predY]
                        )
                        del predY;
                    #################################
                    # augmented input: predict with quantile-selected masked activations
                    for i_alpha, alpha in enumerate(alphas):
                        par_list[0] = alpha
                        for i_perc, perc in enumerate(percentiles):
                            par_list[1] = perc
                        
                            predY = model(copy_sampleX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                            loc_key = (alpha, perc, aug_name2)
                            y_pred_quantile_mask[aug_name][loc_key] = (
                                y_pred_quantile_mask[aug_name].get(loc_key, []) + [predY]
                            )
                            del predY;
                    ###############################
                del copy_sampleX;
                
            del sampleX;
            if (i_sample < Nsamples-1) and (n_samplesX % batch_size != 0):
                continue
                    
            total_samplesX += n_samplesX
            info_msg = f'sampled={total_samplesX}/{Nsamples}'
            if verbose:
                print(info_msg, end='\r')
            if log_path is not None:
                logging.info(info_msg)
                
            # non-augmented input: non-masked predictions
            partX = torch.stack(partX, dim=0)
            predY = model(partX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
            y_pred_no_mask[no_aug_key] = y_pred_no_mask.get(no_aug_key, []) + [predY]
            del predY;
            #####################################
            aug_offset = 0
            y_pred_full_mask[no_aug_key] = y_pred_full_mask.get(no_aug_key, {})
            y_pred_quantile_mask[no_aug_key] = y_pred_quantile_mask.get(no_aug_key, {})
            for augmentation_set_number in augmentation_set_numbers_list:
                n_cur_aug = len(augmentation_names_dict[augmentation_set_number])
                for i_aug2 in range(n_cur_aug):
                    par_list[2] = aug_offset + i_aug2
                    aug_name2 = augmentation_names_dict[augmentation_set_number][i_aug2]
                    if i_aug2 >= n_cur_aug-N_aux:
                        aug_name2 = f'{aug_name2}-{augmentation_set_number}'
                    # non-augmented input: predict with full-sized masked activations
                    for invert in [False, True]:
                        par_list[3] = invert
                        predY = model(partX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                        loc_key = (aug_name2, invert)
                        y_pred_full_mask[no_aug_key][loc_key] = (
                            y_pred_full_mask[no_aug_key].get(loc_key, []) + [predY]
                        )
                        del predY;
                    #################################
                    # augmented input: predict with quantile-selected masked activations
                    for i_alpha, alpha in enumerate(alphas):
                        par_list[0] = alpha
                        for i_perc, perc in enumerate(percentiles):
                            par_list[1] = perc
                        
                            predY = model(partX.to(device), **inference_params).to(device='cpu')#.detach().numpy()
                            loc_key = (alpha, perc, aug_name2)
                            y_pred_quantile_mask[no_aug_key][loc_key] = (
                                y_pred_quantile_mask[no_aug_key].get(loc_key, []) + [predY]
                            )
                            del predY;
                    ###############################
                aug_offset += n_cur_aug
            #################################
            
            # y_true
            y_true = torch.from_numpy(np.array(y_true))
            save_predictions_to_hdf5(
                save_path,
                torch.cat([y_true], dim=0),
                group_name=y_true_key,
                top_n=top_n,
                alpha=None,
                percentile=None,
                aug_name=None,
                invert=None,
            )
            ###################################
            for aug_key in y_pred_quantile_mask:
                for alpha_perc_aug in y_pred_quantile_mask[aug_key]:
                    alpha, perc, aug_name2 = alpha_perc_aug
                    #print(torch.stack(y_pred_quantile_mask[aug_key][alpha_perc_aug], dim=0).shape)
                    #print(y_pred_quantile_mask[aug_key][alpha_perc_aug][0].shape)
                    #print(len(y_pred_quantile_mask[aug_key][alpha_perc_aug]))
                    cur_y = y_pred_quantile_mask[aug_key][alpha_perc_aug]
                    if aug_key == no_aug_key:
                        cur_y = torch.cat(cur_y, dim=0)
                    else:
                        cur_y = torch.stack(cur_y, dim=0)
                    save_predictions_to_hdf5(
                        save_path,
                        cur_y,
                        group_name=aug_key,
                        top_n=top_n,
                        alpha=alpha,
                        percentile=perc,
                        aug_name=aug_name2,
                        invert=None,
                    )
                    del cur_y;
            for aug_key in y_pred_full_mask:
                for aug_invert in y_pred_full_mask[aug_key]:
                    aug_name2, invert = aug_invert
                    cur_y = y_pred_full_mask[aug_key][aug_invert]
                    if aug_key == no_aug_key:
                        cur_y = torch.cat(cur_y, dim=0)
                    else:
                        cur_y = torch.stack(cur_y, dim=0)
                    save_predictions_to_hdf5(
                        save_path,
                        cur_y,
                        group_name=aug_key,
                        top_n=top_n,
                        alpha=None,
                        percentile=None,
                        aug_name=aug_name2,
                        invert=invert,
                    )
                    del cur_y;
            for aug_key in y_pred_no_mask:
                cur_y = y_pred_no_mask[aug_key]
                if aug_key == no_aug_key:
                    cur_y = torch.cat(cur_y, dim=0)
                else:
                    cur_y = torch.stack(cur_y, dim=0)
                save_predictions_to_hdf5(
                    save_path,
                    cur_y,
                    group_name=aug_key,
                    top_n=top_n,
                    alpha=None,
                    percentile=None,
                    aug_name=no_aug_key,
                    invert=None,
                )
                del cur_y;
            del partX, y_pred_quantile_mask, y_true, y_pred_full_mask, y_pred_no_mask;
            partX, y_true = [], []
            y_pred_quantile_mask = {}
            y_pred_full_mask = {}
            y_pred_no_mask = {}
            n_samplesX = 0
    torch_utils.remove_all_hooks_by_dict(handlers)
    info_msg = f'Finished.'
    if verbose:
        print(f'\n{info_msg}')
    if log_path is not None:
        logging.info(info_msg)
    if trace_sampled_params_path is not None:
        np.savez_compressed(trace_sampled_params_path, **traced_params)
    return save_path
    
''';


def extract_accuracy(
    results_path,
    no_aug_key='original',
    y_true_key='true_labels',
    label_name='labels',
    predictions_name='predictions',
    verbose=True,
    Nsamples_max=None,
):
    results_top1 = {}
    results_topn = {}

    with h5py.File(results_path, 'r') as hf:
        augmentation_names = list(hf.keys())
        # print(keys)
        augmentation_names.remove(y_true_key)
        y_true = np.array(hf[y_true_key][label_name])
        Nsamples = Nsamples_selected = y_true.size
        if Nsamples_max is not None:
            Nsamples_selected = min(Nsamples_max, Nsamples)
        y_true = y_true[:Nsamples_selected]
        uni_classes, uni_classes_cnts = np.unique(y_true, return_counts=True)
        print(
            f'Nsamples={Nsamples_selected}: num.classes={len(uni_classes)}, '
            f'min/mean/max samples per class={uni_classes_cnts.min()}/{uni_classes_cnts.mean():.2f}/{uni_classes_cnts.max()}')
        # print(Nsamples)
        y_true = y_true.reshape((Nsamples_selected, 1))
        for augmentation_input_name in augmentation_names:
            for augmentation_mask_config_str in hf[augmentation_input_name]:
                current_group = hf[augmentation_input_name][augmentation_mask_config_str]
                #shape = [Nsamples, -1] + list(hf[x][y].attrs['shape'])
                #shape = [Nsamples, -1, 5]
                n_preds = current_group.attrs['shape'][-1]
                shape = (Nsamples, -1, n_preds)
                y_pred = np.array(current_group[predictions_name]).reshape(shape)
                y_pred = y_pred[:Nsamples_selected]
                #shapes.add(tuple(shape))
                
                acc_top_1 = np.mean(y_pred[:, :, 0] == y_true, axis=0)
                acc_top_1 = np.mean(acc_top_1)
                
                acc_top_n = np.mean(
                    np.isclose(y_pred - y_true[:, :, None], 0).any(axis=-1), axis=-1
                )
                acc_top_n = np.mean(acc_top_n)
                
                results_top1[
                    f'{augmentation_input_name}::{augmentation_mask_config_str}'
                ] = acc_top_1
                results_topn[
                    f'{augmentation_input_name}::{augmentation_mask_config_str}'
                ] = acc_top_n
                if verbose: 
                    augmentation_mask_name = augmentation_mask_config_str.split('=')[1]
                    if (augmentation_input_name == augmentation_mask_name == no_aug_key):
                        print(
                            f'{augmentation_input_name}::{augmentation_mask_config_str} '
                            f'top-1 acc={acc_top_1:.2f}'
                        )
                        print(
                            f'{augmentation_input_name}::{augmentation_mask_config_str} '
                            f'top-{n_preds} acc={acc_top_n:.2f}'
                        )
    return results_top1, results_topn

def collect_featured_measurements(
    accuracy_results_dict,
    augmentation_set_numbers_list,
    alphas,
    percentiles,
    inverts,
    no_aug_key='original',
):
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=False,
    )
    augmentation_and_auxilliary_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=True,
    )
    auxilliary_names_dict = dict(
        (
            aug_set_num,
            list(filter(lambda x: x not in augmentation_names_dict[aug_set_num], names))
        ) for aug_set_num, names in augmentation_and_auxilliary_names_dict.items()
    )
    
    n_features = len(alphas)*len(percentiles) + len(inverts)
    alpha_perc_feature_keys = []
    for alpha, perc in itertools.product(alphas, percentiles):
        alpha_perc_feature_keys.append(f'alpha={alpha:.2f}_perc={perc:.2f}')
    invert_feature_keys = []
    for invert in inverts:
        invert_feature_keys.append(f'invert={invert}')
    
    featured_measurements = {}
    featured_measurements_no_aug_input = {}
    measurements_no_mask = {}
    
    
    for aug_set in augmentation_set_numbers_list:
        aug_names = augmentation_names_dict[aug_set]
        aux_names = list(map(lambda x: f'{x}-{aug_set}', auxilliary_names_dict[aug_set]))
        
        N_aug = len(aug_names)
        N_aux = len(aux_names)
        
        fm = np.zeros((N_aug, N_aug+N_aux, n_features))
        fm_nai = np.zeros((N_aug+N_aux, n_features))
        m_nm = np.zeros((N_aug+1, ))
        
        m_nm[-1] = accuracy_results_dict[f'{no_aug_key}::iaug={no_aug_key}']
        for i, aug1 in enumerate(aug_names):
            m_nm[i] = accuracy_results_dict[f'{aug1}::iaug={no_aug_key}']
            for j, aug2 in enumerate(aug_names+aux_names):
                for k, feature_key in enumerate(alpha_perc_feature_keys):
                    fm[i, j, k] = accuracy_results_dict[f'{aug1}::{feature_key}_iaug={aug2}']
                    if i == 0:
                        fm_nai[j, k] = accuracy_results_dict[f'{no_aug_key}::{feature_key}_iaug={aug2}']
                k_offset = len(alpha_perc_feature_keys)
                for k, feature_key in enumerate(invert_feature_keys):
                    fm[i, j, k_offset+k] = accuracy_results_dict[f'{aug1}::iaug={aug2}_{feature_key}']
                    if i == 0:
                        fm_nai[j, k_offset+k] = accuracy_results_dict[f'{no_aug_key}::iaug={aug2}_{feature_key}']
        m_nm = pd.DataFrame(data=m_nm[None, :], columns=aug_names+[no_aug_key])
        featured_measurements[aug_set] = fm
        featured_measurements_no_aug_input[aug_set] = fm_nai
        measurements_no_mask[aug_set] = m_nm
        del fm, fm_nai, m_nm;
    return featured_measurements, featured_measurements_no_aug_input, measurements_no_mask
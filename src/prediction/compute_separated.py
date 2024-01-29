import logging
import os
import shutil
import h5py

import numpy as np
import torch

import torch_utils
import torch_learning

#import sensitivity_analysis.utils
import sensitivity_analysis.contrast

import preparation.single_unit
#import preparation.visualize

import data_loader.utils

from . import compute

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
    save_dirname,
    save_filename_base,
    top_n=5,
    log_filename_base='modified_activations_pred',
    pre_processing_functions=None,
    post_processing_functions=None,
    verbose=True,
    device='cpu',
    no_aug_key = 'original',
    y_true_key = 'true_labels',
):
    '''
    class variable is always the last one; group name and variable name for
    class are supposed to be identical
    '''
    augmentation_and_auxiliary_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=True,
    )
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=False,
    )
    
    if log_filename_base is not None:
        fnms = os.listdir(save_dirname)
        fnms = list(filter(lambda x: x.endswith('.log'), fnms))
        n_restarts = len(fnms)
        log_path = os.path.join(save_dirname, f'{log_filename_base}_{n_restarts}.log')
        torch_learning.initialize_logging(log_path)
    
    alpha, perc, i_aug2, invert = None, None, None, None # dirty hack (global variable)
    par_list = [alpha, perc, i_aug2, invert]
    modify_activations_function = lambda name, mod, inputs, outputs: compute.modify_activations(
        par_list, sensitivity_values_dict, name, mod, inputs, outputs
    )
    handlers = torch_utils.register_forward_hook_by_name(
        model, modify_activations_function, module_names
    )
    Naugs = sampler.num_augs
    print(Naugs, sampler.augs)
    #N_aux = 3 # hardcoded
    #caps = CustomAugParamSampler(problem, augpar_sampler_seeds)
    
    model.eval();
    model = model.to(device)
    partX, y_true = [], []
    y_pred_quantile_mask = {}
    y_pred_full_mask = {}
    y_pred_no_mask = {}
    
    info_msg = 'Started.'
    if log_path is not None:
        logging.info(info_msg)

    sample_v_path = os.path.join(save_dirname, 'sampleV.pkl')
    ind_x_path = os.path.join(save_dirname, 'indX.pkl')
    n_computed_path = os.path.join(save_dirname, 'n_computed.pkl')

    if os.path.isfile(sample_v_path) and os.path.isfile(ind_x_path):
        sampleV = data_loader.utils.load_pickled_data(sample_v_path)
        indX = data_loader.utils.load_pickled_data(ind_x_path)
        n_computed = data_loader.utils.load_pickled_data(n_computed_path)
    else:
        # sample all parameters in advance
        sampleV, indX = sampler.sample(Nouter=Nsamples, Ninner=Ninner_samples)
        n_computed = 0
        data_loader.utils.save_pickled_data(sampleV, sample_v_path)
        data_loader.utils.save_pickled_data(indX, ind_x_path)
        data_loader.utils.save_pickled_data(n_computed, n_computed_path)

    N_samples_left = Nsamples - n_computed
    control_sampleV = sampleV
    del sampleV;
    with torch.no_grad():
        while N_samples_left > 0:
            cur_batch_size = min(batch_size, N_samples_left)
            save_filename = f'{save_filename_base}_{n_computed}--{n_computed+cur_batch_size}.hdf5'
            save_hdf5_path = os.path.join(save_dirname, save_filename)

            sampleV = control_sampleV

            c_indX = indX[n_computed:n_computed+cur_batch_size]
            c_sampleV = sampleV[n_computed:n_computed+cur_batch_size]
            del sampleV;
            sampleX = []
            for ind in c_indX:
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
                    L = len(augmentation_names_dict[augmentation_set_number])# - N_aux
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
                    aug_params = c_sampleV[i][aug_name]
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
                n_cur_aug = len(augmentation_and_auxiliary_names_dict[augmentation_set_number])
                for i_aug2 in range(n_cur_aug):
                    par_list[0] = par_list[1] = None
                    par_list[2] = aug_offset + i_aug2
                    aug_name2 = augmentation_and_auxiliary_names_dict[augmentation_set_number][i_aug2]
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
            
            aug_offset = 0
            y_pred_full_mask[no_aug_key] = y_pred_full_mask.get(no_aug_key, {})
            y_pred_quantile_mask[no_aug_key] = y_pred_quantile_mask.get(no_aug_key, {})
            for augmentation_set_number in augmentation_set_numbers_list:
                n_cur_aug = len(augmentation_and_auxiliary_names_dict[augmentation_set_number])
                for i_aug2 in range(n_cur_aug):
                    par_list[0] = par_list[1] = None
                    par_list[2] = aug_offset + i_aug2
                    aug_name2 = augmentation_and_auxiliary_names_dict[augmentation_set_number][i_aug2]
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
            compute.save_predictions_to_hdf5(
                save_hdf5_path,
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
                    compute.save_predictions_to_hdf5(
                        save_hdf5_path,
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
                    compute.save_predictions_to_hdf5(
                        save_hdf5_path,
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
                compute.save_predictions_to_hdf5(
                    save_hdf5_path,
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
            
            info_msg = f'sampled={n_computed+cur_batch_size}/{Nsamples} [{n_computed}--{n_computed+cur_batch_size}]'
            if verbose:
                print(info_msg, end='\r')
            if log_path is not None:
                logging.info(info_msg)

            n_computed += cur_batch_size
            data_loader.utils.save_pickled_data(n_computed, n_computed_path)
            
    torch_utils.remove_all_hooks_by_dict(handlers)
    info_msg = f'Finished.'
    if verbose:
        print(f'\n{info_msg}')
    if log_path is not None:
        logging.info(info_msg)
    
    return

def copy_predictions_from_hdf5_to_hdf5(
    hdf5_path_source,
    hdf5_path_destination,
    y_true_key='true_labels',
    label_name='labels',
    predictions_name='predictions',
    probas_name='probas',
):
    with h5py.File(hdf5_path_destination, 'a') as fd_to:
        with h5py.File(hdf5_path_source, 'r') as fd_from:
            for group_name in fd_from:
                current_group_from = fd_from[group_name]
                if group_name not in fd_to:
                    current_group_to = fd_to.create_group(group_name)
                    if group_name == y_true_key:
                        current_group_to.attrs['shape'] = current_group_from.attrs['shape']
                        y = current_group_from[label_name]
                        compute._create_data_record(current_group_to, y, val_name=label_name)
                        continue
                else:
                    current_group_to = fd_to[group_name]
                    if group_name == y_true_key:
                        y = current_group_from[label_name]
                        compute._add_data(current_group_to, y, val_name=label_name)
                        continue
                for subgroup_name in current_group_from:
                    current_subgroup_from = current_group_from[subgroup_name]
                    preds = current_subgroup_from[predictions_name]
                    probas = current_subgroup_from[probas_name]
                    if subgroup_name not in current_group_to:
                        current_subgroup_to = current_group_to.create_group(subgroup_name)
                        compute._create_data_record(current_subgroup_to, preds, val_name=predictions_name)
                        compute._create_data_record(current_subgroup_to, probas, val_name=probas_name)
                        current_subgroup_to.attrs['shape'] = current_subgroup_from.attrs['shape']
                    else:
                        current_subgroup_to = current_group_to[subgroup_name]
                        compute._add_data(current_subgroup_to, preds, val_name=predictions_name)
                        compute._add_data(current_subgroup_to, probas, val_name=probas_name)
                    

def gather_results(
    results_dirname_path,
    results_filename_base,
    results_dirname_path2=None,
    y_true_key='true_labels',
    label_name='labels',
    predictions_name='predictions',
    probas_name='probas',
):
    fnms = os.listdir(results_dirname_path)
    fnms = list(
        filter(
            lambda x: x.startswith(results_filename_base) and x.endswith('.hdf5'),
            fnms,
        )
    )
    starts_n_computed = []
    ends_n_computed = []
    for fnm in fnms:
        tmp = fnm.split(results_filename_base)[1]
        tmp = tmp.split('.hdf5')[0]
        tmp = tmp[1:] # remove '_'
        x0, x1 = tmp.split('--')
        starts_n_computed.append(int(x0))
        ends_n_computed.append(int(x1))
    starts_n_computed.sort()
    ends_n_computed.sort()
    assert starts_n_computed[1:] == ends_n_computed[:-1]
    n_computed_vals = starts_n_computed + ends_n_computed[-1:]

    hdf5_path_destination = os.path.join(
        results_dirname_path2 or results_dirname_path,
        f'{results_filename_base}.hdf5',
    )
    for i in range(len(n_computed_vals)-1):
        start, end = n_computed_vals[i:i+2]
        current_results_filename = f'{results_filename_base}_{start}--{end}.hdf5'
        hdf5_path_source = os.path.join(results_dirname_path, current_results_filename)

        copy_predictions_from_hdf5_to_hdf5(
            hdf5_path_source,
            hdf5_path_destination,
            y_true_key=y_true_key,
            label_name=label_name,
            predictions_name=predictions_name,
            probas_name=probas_name,
        )
    return hdf5_path_destination


def gather_all_results(
    network_names,
    sensitivity_value_name_list,
    values_fnm_base,
    output_filename_suffix,
    results_dirname_path,
    resave_output_dir=None,
):
    for net_name in network_names:
        predictions_fnm_base = f'{net_name}_{values_fnm_base}_{output_filename_suffix}'
        for val_name in sensitivity_value_name_list:
            c_dirname_in = os.path.join(
                results_dirname_path,
                net_name,
                val_name,
            )
            results_fnm_base = f'{val_name}_{predictions_fnm_base}_part={dataset_part}'
            print(c_dirname_in, predictions_fnm_base)
            #prediction.compute_separated.gather_results(
            hdf5_path_destination = gather_results(
                results_dirname_path=c_dirname_in,
                results_filename_base=results_fnm_base,
            )
            if resave_output_dir is not None:
                _, resave_output_fnm = os.path.split(hdf5_path_destination)
                shutil.copy2(
                    hdf5_path_destination,
                    os.path.join(resave_output_dir, resave_output_fnm),
                )
    

















        
    

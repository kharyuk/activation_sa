import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.multitest
import scipy.stats

import os
import math
import copy
import collections
import logging

import torch
import torch_utils
import torch_learning

import h5py

import custom_compose
from . import utils

class CustomAugParamSampler(object):
    
    def __init__(
        self,
        problem,
        dataset,
        augpar_sampler_seeds,
        class_sampler_seed=0,
        class_selector_seed=0,
        labels_attr_name='labels'
    ):
        assert 'groups' in problem
        
        self.num_augs = problem['num_augs']
        self.augs = problem['aug_names']
        
        names = problem['names']
        self.num_vars = len(names)
        dists = problem['dists']
        assert len(dists) == self.num_vars
        bounds = problem['bounds']
        assert len(bounds) == self.num_vars
        self.groups = problem['groups']
        assert len(self.groups) == self.num_vars
        self.theta_indices = problem['theta_indices']
        self.aug_types = np.array(problem['aug_types'])
        assert len(self.theta_indices) == (self.aug_types == 'param').sum()+1
        
        self.variables = collections.OrderedDict(
            (
                names[i], {
                    'dist': dists[i],
                    'bounds': bounds[i],
                }
            ) for i in range(self.num_vars)
        )
        
        self.seed(augpar_sampler_seeds)
        
        # special samplers
        assert class_selector_seed is not None, 'Class selector seed must be specified'
        assert class_sampler_seed is not None, 'Class sampler seed must be specified'
        
        # stratified sampling of class indices
        self.class_selector = utils.ClassRandomSelector(
            num_classes=problem['num_classes'], stratified=True, random_seed=class_selector_seed
        )
        
        # sample train/test partition + sample concrete representative of selected class
        targets = np.array(utils.get_dataset_attr(dataset, labels_attr_name))
        if isinstance(dataset, torch.utils.data.Subset):
            indices = np.array(dataset.indices)
            targets = targets[indices]

        self.oc_sampler = utils.OneClassSampler(
            targets,
            random_seed=class_sampler_seed
        )
        
    def seed(self, random_seed):
        n_seeds = self.num_vars
        if n_seeds == 1:
            self.rng = (np.random.default_rng(random_seed), )
            return
        if isinstance(random_seed, int):
            self.init_random_seeds = utils.split_seed(random_seed, n_seeds)
        else:
            self.init_random_seeds = random_seed
        self.rng = tuple(map(np.random.default_rng, self.init_random_seeds))
        
        # self.rngs = [np.random.default_rng(random_seeds[i]) for i in range(self.num_vars)]
    
    def sample(self, Nouter=1, Ninner=1):
        indX = self.class_selector.sample(Nouter)
        samples = []
        indices = []
        for i_outer in range(Nouter):
            indices.append( self.oc_sampler.sample(indX[i_outer]) )
            result = {}
            i_pi = 0
            for i_aug in range(len(self.augs)):
                aug_name = self.augs[i_aug]
                if self.aug_types[i_aug] == 'nonpar':
                    result[aug_name] = None
                    continue
                ind1 = self.theta_indices[i_pi]
                ind2 = self.theta_indices[i_pi+1]
                group_size = ind2-ind1
                a = np.empty((group_size, Ninner))
                for j_vg in range(group_size):
                    vg_name = list(self.variables.keys())[ind1+j_vg]
                    a[j_vg, :] = getattr(self.rng[ind1+j_vg], self.variables[vg_name]['dist'])(
                        *self.variables[vg_name]['bounds'], size=Ninner
                    )
                result[aug_name] = a
                i_pi += 1
            samples.append(result)
        return samples, indices
    
def compute_pm_ranks(x, zero_method='zsplit', axis=0):
    '''
    Adopted from scipy.stats.wilcoxon. sys.settrace() and frame locals exploring
    were too tricky, thus I decided it is better to recompute r_plus/r_minus
    '''
    d = np.asarray(x)
    N = x.shape[axis]
    n_zero = N - np.count_nonzero(d, axis=axis)
    #n_zero = np.sum(d == 0, axis=axis)
    r = scipy.stats.rankdata(np.abs(d), axis=axis)
    r_plus = np.sum((d > 0) * r, axis=axis)
    r_minus = np.sum((d < 0) * r, axis=axis)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=axis)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    return r_plus, r_minus


def rank_biserial_correlation_coefficient(r_plus, r_minus, n):
    assert r_plus.ndim == r_minus.ndim == 1
    T = np.minimum(r_plus, r_minus)
    rv = 4*np.abs(T - 0.5*(r_plus + r_minus))
    rv /= n*(n+1)
    return rv

def compute_rank_biserial_corrcoef(x):
    n = len(x) # n_samples
    r_plus, r_minus = compute_pm_ranks(x, zero_method='zsplit', axis=0)
    return rank_biserial_correlation_coefficient(r_plus, r_minus, n)

def compute_point_biserial_corrcoef(x, y, pbrcc_m0=None, axis=0):
    ys = y.shape
    xs = x.shape
    ny = ys[axis]
    nx = x.shape[axis]
    assert ys[:axis] + ys[axis+1:] == xs[:axis] + xs[axis+1:]
    if pbrcc_m0 is not None:
        assert xs[:axis] + (1, ) + xs[axis+1:] == pbrcc_m0.shape
    else:
        pbrcc_m0 = np.mean(x, axis=axis, keepdims=True)    
    
    pbrcc_m1 = np.mean(y, axis=axis, keepdims=True)
    pbrcc_n01 = nx + ny
    pbrcc_p0 = nx/pbrcc_n01
    pbrcc_p1 = ny/pbrcc_n01
    pbrcc = (pbrcc_m1 - pbrcc_m0) * np.sqrt(pbrcc_p0*pbrcc_p1)
    pbrcc_m01 = pbrcc_p0*pbrcc_m0 + pbrcc_p1*pbrcc_m1
    #del pbrcc_m0, pbrcc_m1
    pbrcc_s01 = np.sum(np.power(x - pbrcc_m01, 2), axis=axis, keepdims=True)
    pbrcc_s01 += np.sum(np.power(y - pbrcc_m01, 2), axis=axis, keepdims=True)
    pbrcc /= np.sqrt(1e-20 + pbrcc_s01/pbrcc_n01)
    
    return np.squeeze(pbrcc, axis=axis)
    
def multiple_tests_corection(pvalues_dict, alpha=0.05, method='fdr_by'):
    pvals = np.empty(0)
    for key in pvalues_dict:
        pvals = np.append(pvals, pvalues_dict[key].flatten())
    corrected_pvals = statsmodels.stats.multitest.multipletests(
        pvals, alpha=alpha, method='fdr_by', is_sorted=False, returnsorted=False
    )
    corrected_pvals = {}
    ind = 0
    for key in pvalues_dict:
        current_shape = pvalues_dict[key].shape
        offset = int(np.prod(current_shape))
        corrected_pvals[key] = np.reshape(pvals[ind:ind+offset], current_shape)
        ind += offset
    return corrected_pvals

def sample_transform(
    input_batch, transform, params#, use_batch_dim_out=True
):
    if params is None:
        return transform(input_batch)
    Nsamples = params.shape[1]
    output_batch = []
    for j in range(Nsamples):
        local_parameters = tuple(params[:, j])            
        local_output_batch = input_batch.clone()
        local_output_batch = transform(local_output_batch, *local_parameters)
        output_batch.append(local_output_batch)
    if Nsamples == 1:
        return output_batch[0]
    if input_batch.dim() == 3:
        return torch.stack(output_batch, dim=0)
    return torch.cat(output_batch)


def compute_activations_wrt_contrast_stats(
    sampler,
    model,
    module_names,
    transform_functions_list,
    inference_params,
    dataset,
    batch_size,
    Nsamples,
    Ninner_samples,
    #augpar_sampler_seeds,
    #save_output_activations=False,
    activations_dirname=None,
    activations_basename='cs_activations',
    log_path='cs_get_activations.log',
    pre_processing_functions=None,
    post_processing_functions=None,
    #class_variable_name='class',
    verbose=True,
    device='cpu',
    trace_sampled_params_path=None,
    #class_sampler_seed=0,
    #class_selector_seed=0,
    #labels_attr_name='labels'
    custom_save_activations_function=None
):
    '''
    class variable is always the last one; group name and variable name for
    class are supposed to be identical
    '''
    
    if log_path is not None:
        torch_learning.initialize_logging(log_path)
    
    if trace_sampled_params_path is not None:
        traced_params = {}
    
    if custom_save_activations_function is None:
        custom_save_activations_function = utils.save_activations
    
    current_activations_path = None # dirty hack (global variable)
    save_activations_function = lambda name, mod, inputs, outputs: custom_save_activations_function(
        current_activations_path, name, mod, inputs, outputs
    )
    handlers = torch_utils.register_forward_hook_by_name(
        model, save_activations_function, module_names
    )
    Naugs = sampler.num_augs
    
    activations_paths = {}
    
    #caps = CustomAugParamSampler(problem, augpar_sampler_seeds)
    
    model.eval();
    model = model.to(device)
    partX, partY = [], []
    n_samplesX, total_samplesX = 0, 0
    with torch.no_grad():
        for i_sample in range(Nsamples):
            
            sampleV, indX = sampler.sample(Nouter=1, Ninner=Ninner_samples)
            indX = indX[0]
            sampleV = sampleV[0]
            if trace_sampled_params_path is not None:
                traced_params = utils.custom_merge_dicts(traced_params, sampleV)
                traced_params = utils.custom_merge_dicts(traced_params, {'class': indX})
            
            sampleX, _ = dataset[indX]
            if pre_processing_functions is not None:
                sampleX = pre_processing_functions(sampleX)
            if post_processing_functions is not None:
                partX.append(post_processing_functions(sampleX.clone()))
            n_samplesX += 1
            
            for i_aug in range(Naugs):
                aug_name = sampler.augs[i_aug]#problem['aug_names'][i_aug]
                aug_params = sampleV[aug_name]
                #print(aug_name, copy_sampleX.size(), sampleX.size())
                copy_sampleX = sample_transform(
                    sampleX.clone(), transform_functions_list[i_aug], aug_params
                )
                #print(aug_name, copy_sampleX.size(), sampleX.size())
                if post_processing_functions is not None:
                    copy_sampleX = post_processing_functions(copy_sampleX)
                #print(aug_name, copy_sampleX.size(), sampleX.size())
                current_activations_path = os.path.join(
                    activations_dirname, f'{activations_basename}_{aug_name}.hdf5'
                )
                #activations_paths.add(current_activations_path)
                activations_paths[aug_name] = current_activations_path
                if copy_sampleX.dim() == 3:
                    copy_sampleX = copy_sampleX.unsqueeze(0)
                predY = model(copy_sampleX.to(device), **inference_params)#.detach().numpy()
                del copy_sampleX, predY
                
            if (i_sample < Nsamples-1) and (n_samplesX % batch_size != 0):
                del sampleX
                continue
                    
            total_samplesX += n_samplesX
            info_msg = f'sampled={total_samplesX}/{Nsamples}'
            if verbose:
                print(info_msg, end='\r')
            if log_path is not None:
                logging.info(info_msg)
            partX = torch.stack(partX, dim=0)
            current_activations_path = os.path.join(
                activations_dirname, f'{activations_basename}_original.hdf5'
            )
            #activations_paths.add(current_activations_path)
            activations_paths['original'] = current_activations_path
            predY = model(partX.to(device), **inference_params)#.detach().numpy()
            #if save_output_activations:
            #    save_activations_function(
            #        name='model_outputs',
            #        mod=None,
            #        inputs=None,
            #        outputs=predY
            #    )
            del partX, predY
            partX = []
            n_samplesX = 0
    torch_utils.remove_all_hooks_by_dict(handlers)
    info_msg = f'Finished.'
    if verbose:
        print(f'\n{info_msg}')
    if log_path is not None:
        logging.info(info_msg)
    if trace_sampled_params_path is not None:
        #variable_names = list(sampler.variables_names_list)#.keys())
        #traced_params = dict((variable_names[key], np.array(value)) for key, value in traced_params.iteritems())
        np.savez_compressed(trace_sampled_params_path, **traced_params)
    return activations_paths

def compute_paired_wilcoxon(
    module_names,
    activations_original_path,
    activations_augmented_path,
    Ninner_samples=10,
    log_path='glance_at_cs.log',
    cs_path='contrast_stats.hdf5',
    Njobs=2,
    buffer_size=1000,
    verbose=True,
    rewrite_hdf5=False,
    tmp_filename_base='tmp_parallel_cs',
    tmp_save_path='./tmp/',
    activations_key='activations'
    #use_slurm=False
):
    #assert Njobs > 1, "non-parallel version is not implemented, increase the Njobs value"
    
        
    #if use_slurm:
    #    # https://slurm.schedmd.com/job_array.html
    #    # os.environ["SLURM_JOB_ID"] # <- should be taken into account if there are concurrent jobs
    #
    #    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    #    min_task_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
    #    max_task_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
    value_names = ['cs_pvals', 'cs_stats', 'cs_rbscc', 'cs_pbrcc']
    #if not use_slurm or (task_id == min_task_id):
    if log_path is not None:
        torch_learning.initialize_logging(log_path)

    info_msg = f'Starting...'
    if verbose:
        print(info_msg)
    if log_path is not None:
        logging.info(info_msg)
    if os.path.isfile(cs_path):
        if rewrite_hdf5:
            os.remove(cs_path)
        else:
            raise RuntimeError(f'File {cs_path} already exists; rewrite_hdf5={rewrite_hdf5}')
    ####

    for module_name in module_names:
        #if not use_slurm or task_id == min_task_id:
        info_msg = f'Processing {module_name}...'
        if verbose:
            print(info_msg, end='\r')
        if log_path is not None:
            logging.info(info_msg)
        ####
        with h5py.File(activations_original_path, 'r') as activ:
            module_group = activ[module_name]
            activations_original = module_group[activations_key]
            activations_original_shape = module_group.attrs['shape']
            Nsamples_orig = activations_original.shape[0]
            Nactivations = activations_original.shape[-1]
            assert Nactivations == int(np.prod(activations_original_shape))
        with h5py.File(activations_augmented_path, 'r') as activ:
            module_group = activ[module_name]
            activations_augmented = module_group[activations_key]
            activations_augmented_shape = module_group.attrs['shape']
            Nsamples_aug = activations_augmented.shape[0]
            #try:
            assert (
                (Nsamples_aug == Nsamples_orig) or
                (Ninner_samples == (Nsamples_aug // Nsamples_orig))
            )
            assert Nactivations == int(np.prod(activations_augmented_shape))
            #except:
            #    if (Nsamples_aug / Nsamples_orig) > Ninner_samples:
            #        print('Nsamples_orig is smaller; preferable to use it')
        
        
        with h5py.File(activations_original_path, 'r') as activ:
            module_group = activ[module_name]
            activations_original = module_group[activations_key]
            with h5py.File(activations_augmented_path, 'r') as activ2:
                module_group = activ2[module_name]
                activations_augmented = module_group[activations_key]
                if Njobs > 1:
                    value_arrays = parallel_process_cs(
                        activations_original,
                        activations_augmented,
                        value_names,
                        Njobs=Njobs,
                        buffer_size=buffer_size,
                        filename_base=tmp_filename_base,
                        save_path=tmp_save_path,
                        #use_slurm=use_slurm
                        #dtype='float32'
                    )
                else:
                    value_arrays = serial_process_cs(
                        activations_original,
                        activations_augmented,
                        value_names,
                        buffer_size=buffer_size
                    )
        #if use_slurm and task_id != min_task_id:
        #    continue
        with h5py.File(cs_path, 'a') as contrast_stats_hdf5:
            module_group_hdf5_dataset = contrast_stats_hdf5.create_group(module_name)
            for name in value_names:
                module_group_hdf5_dataset.create_dataset(
                    name, data=value_arrays[name], compression="gzip", chunks=True,
                    #maxshape=si_array_shape[:2] + (None, ) + si_array_shape[2:]
                )
    #if use_slurm and task_id != min_task_id:
    #    return ''
    utils.copy_attrs_hdf5(activations_augmented_path, cs_path)
    info_msg = f'Finished.'
    if verbose:
        print(info_msg)
    if log_path is not None:
        logging.info(info_msg)
    return cs_path

def parallel_process_cs(
    module_activations,
    module_activations2,
    value_names,
    Njobs=2,
    buffer_size=1000,
    filename_base='tmp_parallel_cs',
    save_path='./tmp/',
    #use_slurm=False
    #dtype='float32'
):
    assert Njobs > 1
    value_arrays = utils.launch_parallel_work(
        Njobs, compute_contrast_stats_of_activations, module_activations, value_names,
        buffer_size=buffer_size, filename_base=filename_base,
        save_path=save_path, class_variable=True, activations2=module_activations2,
        vectorized_target=True, #use_slurm=use_slurm
    )
    return value_arrays

    

def serial_process_cs(
    module_activations,
    module_activations2,
    value_names,
    buffer_size=1000#, filename_base='tmp_serial_cs', save_path='./tmp/',
    #dtype='float32'
):
    value_arrays = utils.launch_serial_work(
        compute_contrast_stats_of_activations, module_activations, value_names, buffer_size=buffer_size,
        class_variable=True, n_verbose=10, activations2=module_activations2, vectorized_target=True
    )
    return value_arrays


def compute_contrast_stats_of_activations(activations_original, activations_augmented):
    Nsamples_aug = activations_augmented.shape[0]
    Nsamples_orig = activations_original.shape[0]
    Ninner_samples = Nsamples_aug // Nsamples_orig
    
    #if Nsamples_aug / Nsamples_orig >= Ninner_samples:
    #    Nsamples_aug = Ninner_samples*Nsamples_orig
    #    activations_augmented = activations_augmented[:Nsamples_aug]
    #else:
    #    raise ValueError
    
    pbrcc = compute_point_biserial_corrcoef(
        activations_original, activations_augmented, pbrcc_m0=None, axis=0
    )
    if Nsamples_aug != Nsamples_orig:
        for i in range(Ninner_samples):
            activations_augmented[i::Ninner_samples] -= activations_original
    else:
        activations_augmented -= activations_original
    
    current_res = scipy.stats.wilcoxon(
        activations_augmented,
        y=None,
        zero_method='zsplit',#'wilcox',
        correction=True,
        alternative='two-sided',
        mode='approx',
        axis=0
    )
    rbscc = compute_rank_biserial_corrcoef(
        activations_augmented
    )
    return [current_res.pvalue, current_res.statistic, pbrcc, rbscc]







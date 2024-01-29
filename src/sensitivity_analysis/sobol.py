#import sys
#sys.path.append('../src')
import os
import logging
import multiprocessing
import collections
import math

import h5py

import torch_learning
import torch_utils

import torch
import numpy as np
import SALib.sample.saltelli

import custom_compose
from . import utils
from . import salib_analyze_sobol_fixed_vect as salib_analyze_sobol_fixed

import software_setup

_sobol_output_values_names_list = [
    'S1', 'S1_conf', 'ST', 'ST_conf'
]
_sobol_output_values_names_list_selected = [
    'S1', 'ST'
]
_sobol_output_values_2d_names_list = ['S2', 'S2_conf']
_sobol_output_values_2d_names_list_selected = ['S2']

class_variable_name = utils._class_variable_name
permutation_variable_name = utils._permutation_variable_name
partition_variable_name = utils._partition_variable_name



class CustomSampler(object):
    
    def __init__(
        self,
        dataset,
        problem,
        compute_s2=False,
        skip_values=0,
        #random_seed=0,
        class_sampler_seed=None,
        #class_selector_seed=None,
        #use_permutation_variable=True,
        #permutation_random_seed=None,
        #partition_sampler_seed=None,
        use_switch_variables=False,
        labels_attr_name='labels'
    ):
        
        #self.Nvariables = problem['num_vars']
        assert 'groups' in problem
        self.Nvariables = problem['num_vars']
        self.Ngroups = len(set(problem['groups']))
        self.compute_s2 = compute_s2
        self.skip_values = skip_values
        self.problem = problem
        
        self.use_switch_variables_indices = None
        if use_switch_variables:
            self.use_switch_variables_indices = utils.extract_switch_variables_indices(
                problem
            )
        
        offset = 0
        
        # e.g., train / valid / test
        self.use_partition_variable = permutation_variable_name in self.problem['names']
        if self.use_partition_variable:
            assert isinstance(dataset, (list, tuple))
            self.n_partitions = len(dataset)
            offset += 1
        else:
            self.n_partitions = 1
        
        # classes
        # stratified sampling of class indices
        self.use_class_variable = class_variable_name in self.problem['names']
        assert class_sampler_seed is not None, 'Class sampler seed must be specified'
        if self.use_partition_variable:
            assert self.n_partitions == 2, f'Only bi-partitioned structure is supported'
            #self.partition_sampler = utils.PartitionSampler(
            #    n_partitions, p=p_test_partition, random_seed=partition_sampler_seed
            #)
            if isinstance(class_sampler_seed, int):
                class_sampler_seed = utils.split_seed(class_sampler_seed, self.n_partitions)
            self.oc_sampler = []
            for i_part in range(self.n_partitions):
                targets = np.array(utils.get_dataset_attr(dataset[i_part], labels_attr_name))
                if isinstance(dataset[i_part], torch.utils.data.Subset):
                    indices = np.array(dataset[i_part].indices)
                    targets = targets[indices]
                self.oc_sampler.append(
                    utils.OneClassSampler(
                        targets,
                        random_seed=class_sampler_seed[i_part]
                    )
                )
            self.num_classes = len(utils.get_dataset_attr(dataset[0], attr_name='classes'))
        else:
            targets = np.array(utils.get_dataset_attr(dataset, labels_attr_name))
            if isinstance(dataset, torch.utils.data.Subset):
                indices = np.array(dataset.indices)
                targets = targets[indices]
            self.oc_sampler = utils.OneClassSampler(
                targets,
                random_seed=class_sampler_seed
            )
            self.num_classes = len(utils.get_dataset_attr(dataset, attr_name='classes'))
            
        if self.use_class_variable:
            offset += 1
        #self.class_selector = None
        #if class_selector_seed is not None:
        #    self.class_selector = utils.ClassRandomSelector(
        #        num_classes=self.num_classes, stratified=True, random_seed=class_selector_seed
        #    )
        

        # sample permutation of variables
        #self.permutation_sampler = None
        self.use_permutation_variable = permutation_variable_name in self.problem['names']
        if self.use_permutation_variable:
            offset += 1
            self.max_num_sigmas = math.factorial(self.Ngroups-offset)
            self.group_variables_arange = list(range(self.Ngroups-offset))
            
        #self.random_seed = random_seed   
        #self.seed(random_seed)
        
        
        
        
    
    def sample(self, N):
        '''
        Sample up to every inner variable.
        result depends on the N (e.g., for class variable)
        '''
        
        parameter_values = SALib.sample.saltelli.sample(
            self.problem, N, calc_second_order=self.compute_s2, skip_values=self.skip_values,
        ) # ( Nsamples, Nvariables )
        
        #use_partition_variable = not isinstance(self.oc_sampler, utils.OneClassSampler)
        #use_class_variable = self.class_selector is not None
            
        ind = self.Nvariables
        partition_values = None
        if self.use_partition_variable:
            ind -= 1
            partition_values = parameter_values[:, ind]
            partition_values = utils.convert_uniform_to_int(partition_values, max_int=self.n_partitions)
        
        indX = None
        if self.use_class_variable:
            ind -= 1
            indX = parameter_values[:, ind]
            
            if self.use_partition_variable:
                for partition_ind in range(self.n_partitions):
                    cind = np.where(partition_values == partition_ind)[0]
                    class_ind = utils.convert_uniform_to_int(
                        indX[cind], self.oc_sampler[partition_ind].num_classes
                    ) # okke
                    # workaround:
                    indX[cind] = np.array(
                        list(map(lambda x: self.oc_sampler[partition_ind].sample(x), class_ind))
                    )
            else:
                indX = utils.convert_uniform_to_int(
                    indX, self.oc_sampler.num_classes
                ) # okke
                # workaround:
                indX = np.array(
                    list(map(lambda x: self.oc_sampler.sample(x), class_ind))
                )
        else:
            raise NotImplementedError
        indX = indX.astype('i')

        sigmas = None
        if self.use_permutation_variable:
            #sigma = custom_compose.permute_by_index(sigmas[i_sample], transform_variables_arange)
            ind -= 1
            sigmas = parameter_values[:, ind]#.astype('i')
            sigmas = utils.convert_uniform_to_int(sigmas, max_int=self.max_num_sigmas)
            sigmas = list(
                map(lambda pind: custom_compose.permute_by_index(pind, self.group_variables_arange), sigmas)
            )

        return parameter_values[:, :ind], partition_values, indX, sigmas
    
    
    #def __call__(self, group_variable_numbers, N=1, result=None):
    #    return self.sample(group_variable_numbers, N, result)
    

def composed_transform(
    input_batch,
    transform_functions_list,
    sampled_variables,
    theta_indices,
    transforms_thresholds=None,
    use_switch_variables_indices=None,
    permutations=None
):
    '''
    Single input batch, multiple transforms.
    '''
    N_transforms = len(transform_functions_list)
    Nsamples = len(sampled_variables)
    #if N_transforms < len(transform_parameters):

    if transforms_thresholds is None:
        transforms_thresholds = [0.5]*N_transforms
    #paired_tfun_p = list(zip(transform_functions_list, transform_parameters))
    tfun_p = []
    output_batch = []
    for j in range(Nsamples):
        for i in range(N_transforms):
            ind0, ind1 = theta_indices[i], theta_indices[i+1]
            p = None
            if (
                (use_switch_variables_indices is not None) and
                (i in use_switch_variables_indices)
            ):
                p = sampled_variables[j, ind0]
                ind0 += 1
            local_parameters = []
            if ind0 < ind1:
                local_parameters = sampled_variables[j, ind0:ind1]
            local_parameters = tuple(local_parameters)
            tfun_p.append((transform_functions_list[i], p, local_parameters))
        #if permutations is not None:
        #    tfun_p = custom_compose.permute_by_index(permutations[j], tfun_p)
        local_output_batch = input_batch.clone()
        for i_p in range(N_transforms):
            if permutations is None:
                ind = i_p
            else:
                if isinstance(permutations[0], int):
                    ind = permutations[i_p]
                else:
                    ind = permutations[j][i_p]
            # ind = i_p if permutation is None else permutation[i_p]
            transform, p, local_parameters = tfun_p[ind]
            if (p is None) or (p < transforms_thresholds[i]):
                local_output_batch = transform(local_output_batch, *local_parameters)
        #output_batch = input_batch
        #for i, (transform, p, local_parameters) in enumerate(tfun_p):
        #    if (p is None) or :
        #        output_batch = transform(output_batch, *local_parameters)
    #return output_batch
        output_batch.append(local_output_batch)
    if Nsamples == 1:
        return output_batch[0]
    return torch.cat(output_batch)

#@numba.jit
def map_si2array(si, meaningful_indices_2d=None):
    output, output2 = [], []
    for name in _sobol_output_values_names_list_selected:
        output.append(si.get(name, np.nan))
    if meaningful_indices_2d is None:
        return np.array(output), None
    for name in _sobol_output_values_2d_names_list_selected:
        tmp = si.get(name, np.nan)
        output2.append(tmp[meaningful_indices_2d])
    return np.array(output), np.array(output2)

def compute_activations_wrt_sobol_indices(
    sampler,
    model,
    module_names,
    transform_functions_list,
    inference_params,
    dataset,
    batch_size,
    Nsamples,
    #use_switch_variables=False,
    #save_output_activations=False,
    activations_path='activations.hdf5',
    log_path='si_get_activations.log',
    pre_processing_functions=None,
    post_processing_functions=None,
    #use_permutation_variable=True,
    #use_partition_variable=False,
    #class_sampler_seed=0,
    verbose=True,
    device='cpu',
    #p_test_partition=0.5,
    trace_sampled_params_path=None,
    #labels_attr_name='labels',
    custom_save_activations_function=None,
    #handle_save_activations_function=True
):
    if log_path is not None:
        torch_learning.initialize_logging(log_path)
    #num_variables = problem['num_vars']
    
    if custom_save_activations_function is None:
        custom_save_activations_function = utils.save_activations
    save_activations_function = lambda name, mod, inputs, outputs: custom_save_activations_function(
        activations_path, name, mod, inputs, outputs
    )
    #if handle_save_activations_function:
    handlers = torch_utils.register_forward_hook_by_name(
        model, save_activations_function, module_names
    )
    #theta_indices = problem.pop('theta_indices')
    # sample parameters
    
    transform_parameter_values, partition_values, indX, sigmas = sampler.sample(Nsamples)
    if trace_sampled_params_path is not None:
        ind = transform_parameter_values.shape[-1]
        variable_names = list(sampler.problem['names'])#.keys())
        traced_params = dict(
            (variable_names[i], transform_parameter_values[:, i]) for i in range(ind)
        )
        if sigmas is not None:
            traced_params[variable_names[ind]] = sigmas
            ind += 1
        traced_params[variable_names[ind]] = indX
        ind += 1
        if partition_values is not None:
            traced_params[variable_names[ind]] = partition_values
        np.savez_compressed(trace_sampled_params_path, **traced_params)
    
    
    N_transformation_samples = len(transform_parameter_values)
    #batch_size
    model.eval();
    model = model.to(device)
    #param_margin = 1
    #if use_partition_variable:
    #    n_partitions = len(dataset)
    #    assert n_partitions == 2, f'Only bi-partitioned structure is supported'
    #    assert 0 < p_test_partition < 1
    #    param_margin += 1
    #    oc_sampler = []
    #    for i_part in n_partitions:
    #        oc_sampler.append(
    #            utils.OneClassSampler(getattr(dataset[i_part], labels_attr_name), class_sampler_seed)
    #        )
    #else:
    #    oc_sampler = utils.OneClassSampler(getattr(dataset, labels_attr_name), class_sampler_seed)
    partX, partY = [], []
    #np.random.seed(sampler_seed) # never do the global seeding.
    if partition_values is None:
        current_dataset = dataset
    with torch.no_grad():
        for j in range(len(transform_parameter_values)):
            if partition_values is not None:
                current_dataset = dataset[partition_values[j]]
            
            sampleX, _ = current_dataset[indX[j]]
            if pre_processing_functions is not None:
                sampleX = pre_processing_functions(sampleX)
            sampleX = composed_transform(
                sampleX,
                transform_functions_list,
                transform_parameter_values[j:j+1],
                sampler.problem['theta_indices'],
                transforms_thresholds=None,
                use_switch_variables_indices=sampler.use_switch_variables_indices,
                permutations=sigmas[j],
            )
            if post_processing_functions is not None:
                sampleX = post_processing_functions(sampleX)
            partX.append(sampleX)
            if (j < len(transform_parameter_values)-1) and ((j+1) % batch_size != 0):
                del sampleX
                continue
            info_msg = f'params={j+1}/{N_transformation_samples}'
            print(info_msg, end='\r')
            if log_path is not None:
                logging.info(info_msg)
            #partX, partY = torch.cat(partX), torch.cat(partY)
            partX = torch.stack(partX, dim=0)
            predY = model(partX.to(device), **inference_params)#.detach().numpy()
            #if save_output_activations:
            #    save_activations_function(
            #        name='model_outputs',
            #        mod=None,
            #        inputs=None,
            #        outputs=predY
            #    )
            del partX, predY, sampleX
            partX = []
    #if handle_save_activations_function:
    torch_utils.remove_all_hooks_by_dict(handlers)
    info_msg = f'Finished.'
    print(info_msg)
    if log_path is not None:
        logging.info(info_msg)
    return activations_path


def analyze_transform_with_sobol(
    problem,
    module_names,
    activations_path,
    value_names,
    n_proc=8,
    log_path='glance_at_si.log',
    si_path='sobol_indices.hdf5',
    Njobs=1,
    buffer_size=1000,
    seed=123,
    class_variable=False,
    verbose=True,
    rewrite_hdf5=False,
    tmp_filename_base='tmp_parallel_si',
    tmp_save_path='./tmp/',
    activations_key='activations'
    #use_slurm=False
):
    #assert Njobs > 1, "non-parallel version is outdated, increase the Njobs value"
    #if use_slurm:
    #    # https://slurm.schedmd.com/job_array.html
    #    # os.environ["SLURM_JOB_ID"] # <- should be taken into account if there are concurrent jobs
    #
    #    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    #    min_task_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
    #    max_task_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
    #if not use_slurm or (task_id == min_task_id):
    if log_path is not None:
        torch_learning.initialize_logging(log_path)
    info_msg = f'Starting...'
    if verbose:
        print(info_msg)
    if log_path is not None:
        logging.info(info_msg)
    if os.path.isfile(si_path):
        if rewrite_hdf5:
            os.remove(si_path)
        else:
            raise RuntimeError(f'File {si_path} already exists; rewrite_hdf5={rewrite_hdf5}')
    ##########

    for module_name in module_names:
        #if not use_slurm or task_id == min_task_id:
        info_msg = f'Processing {module_name}...'
        if verbose:
            print(info_msg, end='\r')
        if log_path is not None:
            logging.info(info_msg)
        ######
        with h5py.File(activations_path, 'r') as activ:
            module_group = activ[module_name]
            activations = module_group[activations_key]
            if Njobs == 1:
                value_arrays = serial_process_sobol(
                    problem,
                    activations,
                    value_names,
                    n_proc,
                    seed,
                    buffer_size=buffer_size,
                    # dtype='float32',
                    class_variable=class_variable
                )
            else:
                value_arrays = parallel_process_sobol(
                    problem,
                    activations,
                    value_names,
                    n_proc,
                    seed,
                    Njobs=Njobs,
                    buffer_size=buffer_size,
                    filename_base=tmp_filename_base,
                    save_path=tmp_save_path,
                    #dtype='float32',
                    class_variable=class_variable,
                    #use_slurm=use_slurm
                )
        #if use_slurm and task_id != min_task_id:
        #    continue
        with h5py.File(si_path, 'a') as sobol_indices:
            module_group_hdf5_dataset = sobol_indices.create_group(module_name)
            for name in value_names:
                module_group_hdf5_dataset.create_dataset(
                    name, data=value_arrays[name], compression="gzip", chunks=True,
                    #maxshape=si_array_shape[:2] + (None, ) + si_array_shape[2:]
                )
    #if use_slurm and task_id != min_task_id:
    #    return ''
    utils.copy_attrs_hdf5(activations_path, si_path)
    info_msg = f'Finished.'
    if verbose:
        print(info_msg)
    if log_path is not None:
        logging.info(info_msg)
    return si_path


def parallel_process_sobol(
    problem,
    module_activations,
    value_names,
    n_proc,
    seed,
    Njobs=2,
    buffer_size=1000,
    filename_base='tmp_parallel_si',
    save_path='./tmp/',
    dtype='float32',
    class_variable=True,
    #use_slurm=False
):
    assert Njobs > 1
    compute_s2 = 'si2' in value_names
    num_variables = problem['num_vars']
    if 'groups' in problem:
        num_variables = len(set(problem['groups']))
    meaningful_indices_2d = None
    if compute_s2:
        meaningful_indices_2d = np.triu_indices(num_variables, k=1)
        N_s2_interactions = (num_variables*(num_variables-1))//2 # (x, y) interactions
        
    def target_function(a_samples):
        #current_si = SALib.analyze.sobol.analyze(
        #current_si = sobol_func(
        current_si = salib_analyze_sobol_fixed.analyze(
            problem, a_samples, calc_second_order=compute_s2, num_resamples=100,
            conf_level=0.95, print_to_console=False, parallel=False, n_processors=n_proc, seed=seed
        )
        tmp0, tmp1 = map_si2array(current_si, meaningful_indices_2d)
        tmp0 = tmp0.astype(dtype)
        if tmp1 is not None:
            tmp1 = tmp1.astype(dtype)
        return tmp0, tmp1
    
    value_arrays = utils.launch_parallel_work(
        Njobs, target_function, module_activations, value_names, buffer_size=buffer_size,
        filename_base=filename_base, save_path=save_path, class_variable=class_variable,
        vectorized_target=True,# use_slurm=use_slurm#False
        #n_verbose=10, n_trials=10, activations2=None, 
    )
    
    return value_arrays

def serial_process_sobol(
    problem,
    module_activations,
    value_names,
    n_proc,
    seed,
    buffer_size=1000,
    dtype='float32',
    class_variable=True
):
    compute_s2 = 'si2' in value_names
    num_variables = problem['num_vars']
    if 'groups' in problem:
        num_variables = len(set(problem['groups']))
    meaningful_indices_2d = None
    if compute_s2:
        meaningful_indices_2d = np.triu_indices(num_variables, k=1)
        N_s2_interactions = (num_variables*(num_variables-1))//2 # (x, y) interactions
        
    def target_function(a_samples):
        #current_si = SALib.analyze.sobol.analyze(
        #current_si = sobol_func(
        current_si = salib_analyze_sobol_fixed.analyze(
            problem, a_samples, calc_second_order=compute_s2, num_resamples=100,
            conf_level=0.95, print_to_console=False, parallel=False, n_processors=n_proc, seed=seed
        )
        tmp0, tmp1 = map_si2array(current_si, meaningful_indices_2d)
        tmp0 = tmp0.astype(dtype)
        if tmp1 is not None:
            tmp1 = tmp1.astype(dtype)
        return tmp0, tmp1
    
    value_arrays = utils.launch_serial_work(
        target_function, module_activations, value_names, buffer_size=buffer_size,
        class_variable=class_variable, #n_verbose=10, activations2=None,
        vectorized_target=True#False
    )
    return value_arrays

















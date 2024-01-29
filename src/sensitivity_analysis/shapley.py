import numpy as np
import matplotlib.pyplot as plt

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

import image_transforms.pre_post_processing
import data_loader.general
import data_loader.imagenet

        

class CustomSampler(object):
    
    def __init__(
        self,
        dataset,
        problem,
        Vi_groups,
        theta_0,
        random_seed=0,
        class_sampler_seed=None,
        class_selector_seed=None,
        permutation_random_seed=None,
        partition_sampler_seed=None,
        #n_partitions=2,
        p_test_partition=0.5,
        labels_attr_name='labels'
    ):
        assert 'groups' in problem
        num_vars = problem['num_vars']
        #names = problem.get('names', [f'v{i}' for i in range(num_vars)])
        self.variables_names_list = problem['names']
        dists = problem.get('dists', ['uniform']*num_vars) # just ignoring in current implementation
        bounds = problem.get('bounds', [(0, 1)]*num_vars) # just ignoring in current implementation
        groups = problem['groups']
        self.variables = collections.OrderedDict(
            (
                self.variables_names_list[i], {
                    'dist': dists[i],
                    'bounds': bounds[i],
                    'group': groups[i]
                }
            ) for i in range(num_vars)
        )
        
        self.theta_indices = problem.get('theta_indices', None)
        if self.theta_indices is None:
            self.theta_indices = utils.extract_group_boundary_indices(groups, num_vars)
        
        
        #uni_groups = []
        #for gname in groups:
        #    if gname in uni_groups:
        #        continue
        #    uni_groups.append(gname)
        
        self.groups = collections.OrderedDict(
            (
                groups[self.theta_indices[i]],
                np.array(self.variables_names_list)[self.theta_indices[i]:self.theta_indices[i+1]]
            ) for i in range(len(self.theta_indices)-1)
        ) # not the best solution but works
        
        
        self.num_main_variables = len(self.groups)
        self.groups_lengthes = dict(
            (group, sum(np.array(groups) == group)) for group in self.groups
        )
        
        assert set(Vi_groups.keys()).issubset(set(self.groups))
        self.Vi_groups = Vi_groups # dict((group_name, proba))
        #assert len(set(Vi_groups)) == len(Vi_probas)
        #self.Vi_probas = Vi_probas
        
        assert set(theta_0.keys()).issubset(set(self.groups))
        #assert len(theta_0) == len(Vi_groups) # empty lists for those without internal parameters
        self.theta_0 = theta_0
        
        #self.rng = np.random.default_rng(random_seed)
        
        self.use_switch_variables_indices = utils.extract_switch_variables_indices(
            problem
        )
        
        # special samplers
        assert class_sampler_seed is not None, 'Class sampler seed must be specified'
        margin = 0
        
        # stratified sampling of class indices
        self.class_selector = None
        self.class_variable_ind = None
        self.use_class_variable = False
        if class_selector_seed is not None:
            self.class_variable_ind = self.variables_names_list.index(utils._class_variable_name)
            self.class_selector = utils.ClassRandomSelector(
                num_classes=problem['num_classes'], stratified=True, random_seed=class_selector_seed
            )
            margin += 1
            self.use_class_variable = True
        
        # sample train/test partition + sample concrete representative of selected class
        self.partition_sampler = None
        self.partition_variable_ind = None
        if partition_sampler_seed is not None:
            self.partition_variable_ind = self.variables_names_list.index(utils._partition_variable_name)
            n_partitions = len(dataset)
            assert n_partitions == 2, f'Only bi-partitioned structure is supported'
            self.partition_sampler = utils.PartitionSampler(
                n_partitions, p=p_test_partition, random_seed=partition_sampler_seed
            )
            if isinstance(class_sampler_seed, int):
                class_sampler_seed = utils.split_seed(class_sampler_seed, n_partitions)
            self.oc_sampler = []
            for i_part in range(n_partitions):
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
            margin += 1
            self.use_partition_variable = True
        else:
            targets = np.array(utils.get_dataset_attr(dataset, labels_attr_name))
            if isinstance(dataset, torch.utils.data.Subset):
                indices = np.array(dataset.indices)
                targets = targets[indices]

            utils.OneClassSampler(
                targets,
                random_seed=class_sampler_seed
            )
            self.use_partition_variable = False

        # sample permutation of variables
        self.permutation_sampler = None
        self.permutation_variable_ind = None
        self.use_permutation_variable = False
        if permutation_random_seed is not None:
            margin += 1
            self.permutation_variable_ind = self.variables_names_list.index(utils._permutation_variable_name)
            self.permutation_sampler = utils.PermutationRandomSampler(
                n_variables=self.num_main_variables-margin, oversample=True, permute_result=True,
                permutation_seed=permutation_random_seed
            )
            self.use_permutation_variable = True
            
        self.seed(random_seed)
        
        
    def seed(self, random_seed):
        n_seeds = len(self.variables_names_list)
        n_seeds -= int(self.permutation_sampler is not None)
        n_seeds -= int(self.partition_sampler is not None)
        n_seeds -= int(self.class_selector is not None)
        if n_seeds == 1:
            self.rng = (np.random.default_rng(random_seed), )
            return
        if isinstance(random_seed, int):
            self.init_random_seeds = utils.split_seed(random_seed, n_seeds)
        else:
            self.init_random_seeds = random_seed
        self.rng = tuple(map(np.random.default_rng, self.init_random_seeds))
    
    ### variale-wise, not group-wise - how?
    #def group_sample(self, group_variable_numbers, N=1, result=None):
    #    '''
    #    Sample up to every group of variables. All groups are permuted, but variables within group
    #    are sampled simultaneously.
    #    '''
    #    if result is None:
    #        result = {}
    #    for i_v in group_variable_numbers:
    #        group_name = list(self.groups.keys())[i_v]
    #        group_size = self.groups_lengthes[group_name]
    #        a = np.empty((group_size, N))
    #        for j_vg in range(group_size):
    #            vg_name = self.groups[group_name][j_vg]
    #            a[j_vg, :] = getattr(self.rng, self.variables[vg_name]['dist'])(
    #                *self.variables[vg_name]['bounds'], size=N
    #            )
    #        if group_name in self.Vi_groups:
    #            #a[0, :] = self.rng.randint(0, 2, size=(N, ))
    #            ind = np.where(a[0] < 1.-self.Vi_groups[group_name])[0]
    #            if len(a) > 1:
    #                a[1:, ind] = self.theta_0[group_name][:, None]
    #            a[0, ind] = 0
    #            ind = np.where(a[0] >= 1.-self.Vi_groups[group_name])[0]
    #            a[0, ind] = 1
    #        result[i_v] = a
    #    return result
    
    def sample(self, variable_numbers, N=1, result=None):
        '''
        Sample up to every inner variable.
        result depends on the N (e.g., for class variable)
        '''
        if result is None:
            result = collections.OrderedDict()
        else:
            assert isinstance(result, collections.OrderedDict)
            for key in result:
                assert len(result[key]) == N
                assert key not in variable_numbers
        for i_v in variable_numbers:
            if i_v in result:
                continue
            if i_v == self.permutation_variable_ind:
                result[i_v] = self.permutation_sampler.sample(N)
                continue
            if i_v == self.partition_variable_ind:
                result[i_v] = self.partition_sampler.sample(N).astype('i')
                continue
            if i_v == self.class_variable_ind:
                result[i_v] = self.class_selector.sample(N)
                continue
            
            
            
            name = self.variables_names_list[i_v]
            a = getattr(self.rng[i_v], self.variables[name]['dist'])(
                *self.variables[name]['bounds'], size=N
            )
            if name in self.Vi_groups:
                sampled_variables = set(result.keys())
                group_variables = set(self.groups[name])
                group_sampled_variables = sampled_variables.intersection(group_variables)
                if len(group_sampled_variables) > 0:
                    # they are from theta0  with 0 probability - just sample ones
                    np.fill(a, 1)
                else:
                    #a[0, :] = self.rng.randint(0, 2, size=(N, ))
                    ind = np.where(a < 1.-self.Vi_groups[name])[0]
                    #if len(result[name]) > 1:
                    #    a[1:, ind] = self.theta_0[group_name][:, None]
                    a[ind] = 0
                    ind = np.where(a >= 1.-self.Vi_groups[name])[0]
                    a[ind] = 1
            else:
                group_name = self.variables[name]['group']
                group_v_ind = self.variables_names_list.index(group_name)
                if group_v_ind in result:
                    b = result[group_v_ind]
                    vg_ind = list(self.groups[group_name]).index(name)
                    ind = np.where(b == 0)[0]
                    if len(ind) > 0:
                        a[ind] = self.theta_0[group_name][vg_ind-1] # assuming that we work using indicator variables
            result[i_v] = a
        if self.partition_sampler is not None:
            if (
                (
                    (self.class_variable_ind in variable_numbers) and
                    (self.partition_variable_ind in result)
                ) or (
                    (self.partition_variable_ind in variable_numbers) and
                    (self.class_variable_ind in result)
                )
            ):
                result[self.class_variable_ind] = list(
                    map(
                        lambda i: self.oc_sampler[
                            result[self.partition_variable_ind][i]
                        ].sample(result[self.class_variable_ind][i]),
                        range(N)
                    )
                )
        else:
            if (self.class_variable_ind in variable_numbers):
                result[self.class_variable_ind] = list(
                    map(
                        lambda x: self.oc_sampler.sample(x),
                        result[self.class_variable_ind]
                    )
                )
        return result
    
    
    #def __call__(self, group_variable_numbers, N=1, result=None):
    #    return self.sample(group_variable_numbers, N, result)

def composed_transform(
    input_batch,
    transform_functions_list,
    sampled_variables,
    theta_indices,
    use_switch_transforms_indices,
    permutations
):
    N_transforms = len(transform_functions_list)
    tfun_p = []
    Nsamples = len(sampled_variables[0])#.shape[1]
    output_batch = []
    for j in range(Nsamples):
        for i in range(N_transforms):
            ind0, ind1 = theta_indices[i], theta_indices[i+1]
            p = None
            #print(j, i, ind0, ind1, sampled_variables)
            if (
                (use_switch_transforms_indices is not None) and
                (i in use_switch_transforms_indices)
            ):
                p = sampled_variables[ind0][j]
                ind0 += 1
            local_parameters = []
            if ind0 < ind1:
                local_parameters = [sampled_variables[k][j] for k in range(ind0, ind1)]
            local_parameters = tuple(local_parameters)
            tfun_p.append((transform_functions_list[i], p, local_parameters))
        local_output_batch = input_batch.clone()
        for i_p in range(N_transforms):
            if permutations is None:
                ind = i_p
            else:
                if isinstance(permutations[0], int):
                    ind = permutations[i_p]
                else:
                    ind = permutations[j][i_p]
            #ind = i_p if permutation is None else permutation[i_p]
            transform, p, local_parameters = tfun_p[ind]
            if (p is None) or (p == 1):
                local_output_batch = transform(local_output_batch, *local_parameters)
        output_batch.append(local_output_batch)
    if Nsamples == 1:
        return output_batch[0]
    return torch.cat(output_batch)

'''
def sample_transform_permutations_old(
    sampler,
    Npermutations,
    permutation_random_seed=0,
    class_variable_name=None
):
    permutation_rng = np.random.default_rng(permutation_random_seed)
    Nvariables = len(sampler.groups)
    #if class_variable_name is None:
    max_num_sigmas = math.factorial(Nvariables)
    #transform_variables_arange = list(range(max_num_sigmas))
    transform_variables_arange = list(range(Nvariables))
    sigmas = permutation_rng.choice(
        max_num_sigmas, size=min(Npermutations, max_num_sigmas), replace=False, shuffle=True
    ) # replace=False is essential here!
    return sigmas, transform_variables_arange
'''

def sample_permutations(
    Nvariables,
    Npermutations,
    permutation_seed=0,
    oversample=False,
    permute_result=True
):
    '''
    abstract permutation
    '''
    permutation_rng = np.random.default_rng(permutation_seed)
    max_num_sigmas = math.factorial(Nvariables)
    ind_arange = list(range(Nvariables))
    sigmas = permutation_rng.choice(
        max_num_sigmas, size=min(Npermutations, max_num_sigmas), replace=False, shuffle=True
    ) # replace=False is essential here!
    n_samples = len(sigmas)
    add_more_samples = oversample and (n_samples == Npermutations)
    Nmax = Npermutations - n_samples
    while add_more_samples:
        new_sigmas = permutation_rng.choice(
            max_num_sigmas, size=min(Nmax, max_num_sigmas), replace=False, shuffle=True
        )
        sigmas = np.append(sigmas, new_sigmas, axis=-1)
        n_samples = len(sigmas)
        assert n_samples <= Npermutations
        Nmax = Npermutation - n_samples
        add_more_samples = (n_samples == Npermutations)
    if permute_result:
        return permutation_rng.permutation(sigmas), ind_arange
    return sigmas, ind_arange
    

'''
def sample_transform_permutations(
    Nvariables
    Npermutations,
    use_class_variable,
    use_partition_variable,
    permutation_random_seed=0
):
    
    assert use_class_variable
    
    permutation_rng = np.random.default_rng(permutation_random_seed)
    
    margin = 0
    margin += int(use_class_variable) + int(use_partition_variable)
    Nvariables = len(sampler.groups)-margin
    #if class_variable_name is None:
    max_num_sigmas = math.factorial(Nvariables)
    #transform_variables_arange = list(range(max_num_sigmas))
    variables_ind_arange = list(range(Nvariables))
    sigmas = permutation_rng.choice(
        max_num_sigmas, size=min(Npermutations, max_num_sigmas), replace=False, shuffle=True
    ) # replace=False is essential here!
    return sigmas, variables_ind_arange
'''

'''
sampler = sensitivity_analysis.shapley.CustomSampler(
    (train_dataset, valid_dataset),
    problem,
    Vi_groups,
    Vtheta0_dict,
    random_seed=0,
    class_sampler_seed=0,
    class_selector_seed=0,
    permutation_random_seed=0,
    partition_sampler_seed=0,
    n_partitions=2,
    p_test_partition=0.5,
    labels_attr_name='targets'
)
'''


def compute_activations_wrt_total_variance(
    sampler,
    model,
    module_names,
    transform_functions_list,
    inference_params,
    dataset,
    batch_size,
    Nsamples,
    #use_switch_transforms_indices,
    #save_output_activations=False,
    activations_path='tv_activations.hdf5',
    log_path='tv_get_activations.log',
    pre_processing_functions=None,
    post_processing_functions=None,
    #class_selector_seed=0,
    #class_sampler_seed=0,
    #permutation_random_seed=0,
    #partition_sampler_seed=0,
    #variable_permutation_random_seed=0,
    #p_test_partition=None,
    verbose=True,
    device='cpu',
    trace_sampled_params_path=None,
    save_activations_function=None,
    handle_save_activations_function=True
    #labels_attr_name='labels'
):
    '''
    Augmented variables order: ..., permutation, class, partition
    
    '''
    #if use_class_variable:
    #    crs = utils.ClassRandomSelector(
    #        num_classes=problem['num_classes'], stratified=True, random_seed=class_selector_seed
    #    )
    #    class_samples = crs.sample(Nsamples)
    #else:
    #    raise NotImplementedError
    #param_margin = sum(map(int, [use_class_variable, use_permutation_variable, use_partition_variable]))
    #assert class_variable_ind == len(sampler.groups)-1
    #oc_sampler = utils.OneClassSampler(getattr(dataset, labels_attr_name), class_sampler_seed)
    
    #if use_partition_variable:
    #    n_partitions = len(dataset)
    #    assert n_partitions == 2, f'Only bi-partitioned structure is supported'
    #    partition_sampler = utils.PartitionSampler(
    #        n_partitions, p=p_test_partition, random_seed=partition_sampler_seed
    #    )
    #    
    #    oc_sampler = []
    #    for i_part in n_partitions:
    #        oc_sampler.append(
    #            utils.OneClassSampler(getattr(dataset[i_part], labels_attr_name), class_sampler_seed)
    #        )
    #else:
    #    oc_sampler = utils.OneClassSampler(getattr(dataset, labels_attr_name), class_sampler_seed)
        
    use_class_variable = sampler.class_selector is not None
    use_permutation_variable = sampler.permutation_sampler is not None
    use_partition_variable = sampler.partition_sampler is not None
    margin = int(use_partition_variable) + int(use_permutation_variable) + int(use_class_variable)
    
    if log_path is not None:
        torch_learning.initialize_logging(log_path)
        
    if trace_sampled_params_path is not None:
        traced_params = {}
        
    if not use_partition_variable:
        current_dataset = dataset
    
    if save_activations_function is None:
        save_activations_function = lambda name, mod, inputs, outputs: utils.save_activations(
            activations_path, name, mod, inputs, outputs
        )
    if handle_save_activations_function:
        handlers = torch_utils.register_forward_hook_by_name(
            model, save_activations_function, module_names
        )
    Ngroup_variables = len(transform_functions_list)
    Nvariables = len(sampler.variables) # -param_margin # exclude perm./class/part.
    variables_ind_arange = np.arange(Nvariables)
    #if use_permutation_variable:
    #    sigmas, transform_variables_arange = sample_permutations(
    #        Ngroup_variables, sampler, Nsamples, permutation_seed=permutation_random_seed,
    #        oversample=True, permute_result=True
    #    )
    #else:
    #    sigma = np.arange(Ngroup_variables)
    model.eval();
    model = model.to(device)
    partX, partY = [], []
    n_samplesX, total_samplesX = 0, 0
    #N_transformation_samples = Npermutations*Nsamples
    sigma = None
    with torch.no_grad():
        for i_sample in range(Nsamples):
            sampleV = sampler.sample(variables_ind_arange)
            if trace_sampled_params_path is not None:
                traced_params = utils.custom_merge_dicts(traced_params, sampleV)
            ind_var = Nvariables
            if use_partition_variable:
                #partition_ind = partition_sampler.sample()
                #int(transform_parameters[-1] < 1-p_test_partition) # two partitions
                #indX = oc_sampler[partition_ind].sample(class_samples[i_sample])
                #sampleX, _ = dataset[partition_ind][indX]
                ind_var -= 1
                #print(sampleV, ind_var)
                current_partition = sampleV.pop(ind_var)[0]
                
                current_dataset = dataset[current_partition]
            #else:
            #    indX = oc_sampler.sample(class_samples[i_sample])
            #    sampleX, _ = dataset[indX]
            # now sample aug.parameters
            #sampleV = sampler.sample(variables_ind_arange)
            if use_class_variable:
                ind_var -= 1
                indX = sampleV.pop(ind_var)[0]#.astype('i')[0]
                sampleX, _ = current_dataset[indX]
            else:
                raise NotImplementedError
                
            if use_permutation_variable:
                #sigma = custom_compose.permute_by_index(sigmas[i_sample], transform_variables_arange)
                ind_var -= 1
                sigma = sampleV.pop(ind_var)[0]
            
            
            
            if pre_processing_functions is not None:
                sampleX = pre_processing_functions(sampleX)
            sampleX = composed_transform(
                sampleX,
                transform_functions_list,
                sampleV,
                sampler.theta_indices,
                sampler.use_switch_variables_indices,
                sigma
            )
            if post_processing_functions is not None:
                sampleX = post_processing_functions(sampleX)
            partX.append(sampleX)
            n_samplesX += 1
            if (i_sample < Nsamples-1) and (n_samplesX % batch_size != 0):
                del sampleX
                continue
            total_samplesX += n_samplesX
            info_msg = f'sampled={total_samplesX}/{Nsamples}'
            if verbose:
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
            del partX, predY
            partX = []
            n_samplesX = 0
    if handle_save_activations_function:
        torch_utils.remove_all_hooks_by_dict(handlers)
    info_msg = f'Finished.'
    if verbose:
        print(f'\n{info_msg}')
    if log_path is not None:
        logging.info(info_msg)
    if trace_sampled_params_path is not None:
        variable_names = list(sampler.variables_names_list)#.keys())
        traced_params = dict((variable_names[key], np.array(value)) for key, value in traced_params.items())
        np.savez_compressed(trace_sampled_params_path, **traced_params)
    return activations_path


def compute_activations_wrt_shapley_effect_sampling(
    sampler,
    model,
    module_names,
    transform_functions_list,
    inference_params,
    dataset,
    batch_size,
    Npermutations,
    Nouter_samples,
    Ninner_samples,
    #use_switch_transforms_indices,
    #save_output_activations=False,
    activations_path='activations.hdf5',
    log_path='shpv_get_activations.log',
    pre_processing_functions=None,
    post_processing_functions=None,
    #use_class_variable=True,
    #use_permutation_variable=True,
    #use_partition_variable=False,
    #class_selector_seed=0,
    #class_sampler_seed=0,
    #permutation_random_seed=0,
    #partition_sampler_seed=0,
    variables_permutation_random_seed=0,
    #p_test_partition=None,
    verbose=True,
    device='cpu',
    trace_sampled_params_path=None,
    #labels_attr_name='labels'
    custom_save_activations_function=None,
    #handle_save_activations_function=True
):
    '''
    class variable is always the last one; group name and variable name for
    class are supposed to be identical
    '''
    
    use_class_variable = sampler.class_selector is not None
    use_permutation_variable = sampler.permutation_sampler is not None
    use_partition_variable = sampler.partition_sampler is not None
                                                 
                                                 
    #if use_class_variable:
    #    crs = utils.ClassRandomSelector(
    #        num_classes=problem['num_classes'], stratified=True, random_seed=class_selector_seed
    #    )
    #    class_variable_ind = sampler.variables_names_list.index('class') # ...hardcoded...
    #    #class_samples = crs.sample(Nsamples)
    #else:
    #    raise NotImplementedError
    #param_margin = sum(map(int, [use_class_variable, use_permutation_variable, use_partition_variable]))
    #assert class_variable_ind == len(sampler.groups)-1
    #oc_sampler = utils.OneClassSampler(getattr(dataset, labels_attr_name), class_sampler_seed)
    #if use_permutation_variable:
    #    permutation_variable_ind = sampler.variables_names_list.index('permutation') # ...hardcoded...
    #
    #if use_partition_variable:
    #    partition_variable_ind = sampler.variables_names_list.index('partition') # ...hardcoded...
    #    n_partitions = len(dataset)
    #    assert n_partitions == 2, f'Only bi-partitioned structure is supported'
    #    partition_sampler = utils.PartitionSampler(
    #        n_partitions, p=p_test_partition, random_seed=partition_sampler_seed
    #    )
    #    
    #    oc_sampler = []
    #    for i_part in n_partitions:
    #        oc_sampler.append(
    #            utils.OneClassSampler(getattr(dataset[i_part], labels_attr_name), class_sampler_seed)
    #        )
    #else:
    #    oc_sampler = utils.OneClassSampler(getattr(dataset, labels_attr_name), class_sampler_seed)
    
    if log_path is not None:
        torch_learning.initialize_logging(log_path)
        
    if trace_sampled_params_path is not None:
        traced_params = {}
        
    if not use_partition_variable:
        current_dataset = dataset
        
    if custom_save_activations_function is None:
        custom_save_activations_function = utils.save_activations
    save_activations_function = lambda name, mod, inputs, outputs: custom_save_activations_function(
        activations_path, name, mod, inputs, outputs
    )
    #if handle_save_activations_function:
    handlers = torch_utils.register_forward_hook_by_name(
        model, save_activations_function, module_names
    )
    #Nvariables = problem['Nvariables'] # should be a number of groups of variables !
    Nvariables = len(sampler.variables) # all variables!
    variables_sigmas, variables_arange = sample_permutations(
        Nvariables, Npermutations, permutation_seed=variables_permutation_random_seed,
        oversample=False, permute_result=True
    )
    Npermutations = len(variables_sigmas)
    
    model.eval();
    model = model.to(device)
    partX, partY = [], []
    n_samplesX, total_samplesX = 0, 0
    N_transformation_samples = Npermutations*Nvariables*Nouter_samples*Ninner_samples
    with torch.no_grad():
        for i_vperm, variables_sigma_ind in enumerate(variables_sigmas):
            #sigma = custom_compose.permute_by_index(sigmas[i_perm], transform_variables_arange)
            #local_sigma = copy.copy(sigma)
            #if class_variable_name is not None:
            #    local_sigma.remove(class_variable_ind)
            #print(variables_sigma_ind, type(variables_sigma_ind))
            if isinstance(variables_sigma_ind, (np.ndarray, list, tuple)):
                variables_sigma = variables_sigma_ind
            else:
                variables_sigma = custom_compose.permute_by_index(variables_sigma_ind, variables_arange)
            for i_variable in range(Nvariables):
                # the last one is just a total variation but the permutation
                # matters!
                # conditioned_var_activations
                for i_outer in range(Nouter_samples):
                    # sample sigma(i_variable) ... sigma(Nvariables-1)
                    #print(variables_sigma, i_variable)
                    sampleV_right = sampler.sample(variables_sigma[i_variable+1:])
                    for i_inner in range(Ninner_samples):
                        
                        sampleV = copy.copy(sampleV_right)
                        sampleV = sampler.sample(variables_sigma[:i_variable+1], result=sampleV)
                        #sampleV = {**sampleV_left, **sampleV_right}
                        if trace_sampled_params_path is not None:
                            traced_params = utils.custom_merge_dicts(traced_params, sampleV)
                        ind = Nvariables
                        if use_partition_variable:
                            #partition_ind = partition_sampler.sample()
                            #int(transform_parameters[-1] < 1-p_test_partition) # two partitions
                            #indX = oc_sampler[partition_ind].sample(class_samples[i_sample])
                            #sampleX, _ = dataset[partition_ind][indX]
                            ind -= 1
                            current_partition = sampleV.pop(ind)[0]
                            current_dataset = dataset[current_partition]
                        #else:
                        #    indX = oc_sampler.sample(class_samples[i_sample])
                        #    sampleX, _ = dataset[indX]
                        # now sample aug.parameters
                        #sampleV = sampler.sample(variables_ind_arange)
                        if use_class_variable:
                            ind -= 1
                            indX = sampleV.pop(ind)[0]           
                            sampleX, _ = current_dataset[indX]
                        else:
                            raise NotImplementedError
                            
                        if use_permutation_variable:
                            #sigma = custom_compose.permute_by_index(sigmas[i_sample], transform_variables_arange)
                            ind -= 1
                            local_sigma = sampleV.pop(ind)[0]
                        
                        
                        #if use_class_variable:
                        #    class_ind = sampleV.pop(class_variable_ind)
                        #else:
                        #    raise NotImplementedError
                        #if use_partition_variable:
                        #    partition_ind = sampleV.pop(partition_variable_ind)
                        #    if use_class_variable:
                        #        indX = oc_sampler[partition_ind].sample(int(class_ind))
                        #        sampleX, _ = dataset[indX]
                        #    else:
                        #        raise NotImplementedError
                        #else:
                        #    if use_class_variable:
                        #        indX = oc_sampler.sample(int(class_ind))
                        #        sampleX, _ = dataset[indX]
                        #    else:
                        #        raise NotImplementedError
                        #if use_permutation_variable:
                        #    permutation_ind = sampleV.pop(permutation_variable_ind)
                            
                        if pre_processing_functions is not None:
                            sampleX = pre_processing_functions(sampleX)
                        sampleX = composed_transform(
                            sampleX,
                            transform_functions_list,
                            sampleV,
                            sampler.theta_indices,
                            sampler.use_switch_variables_indices,
                            local_sigma
                        )
                        if post_processing_functions is not None:
                            sampleX = post_processing_functions(sampleX)
                        partX.append(sampleX)
                        n_samplesX += 1
                        if (
                            (
                                (i_inner < Ninner_samples-1) or
                                (i_outer < Nouter_samples-1) or
                                (i_variable < Nvariables-1) or #(i_variable < Nvariables-2) or
                                (i_vperm < Npermutations-1)
                            ) and (
                                n_samplesX % batch_size != 0
                            )
                        ):
                            del sampleX
                            continue
                        total_samplesX += n_samplesX
                        info_msg = f'sampled={total_samplesX}/{N_transformation_samples}'
                        if verbose:
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
                        del partX, predY
                        partX = []
                        n_samplesX = 0
    #if handle_save_activations_function:
    torch_utils.remove_all_hooks_by_dict(handlers)
    info_msg = f'Finished.'
    with h5py.File(activations_path, 'a') as activ:
        activ.attrs.create(
            'sigmas', data=variables_sigmas
        )
    if verbose:
        print(f'\n{info_msg}')
    if log_path is not None:
        logging.info(info_msg)
    if trace_sampled_params_path is not None:
        variable_names = list(sampler.variables_names_list)#.keys())
        traced_params = dict((variable_names[key], np.array(value)) for key, value in traced_params.items())
        traced_params['variables_sigmas'] = variables_sigmas
        np.savez_compressed(trace_sampled_params_path, **traced_params)
    return activations_path, variables_sigmas

def analyze_transform_with_shapley(
    module_names,
    activations_path,
    #variables_sigmas,
    Nvariables,
    Nouter_samples,
    Ninner_samples,
    log_path='glance_at_shpv.log',
    shpv_path='shapley_values.hdf5',
    Njobs=2,
    buffer_size=1000,
    class_variable=True,
    verbose=True,
    rewrite_hdf5=False,
    tmp_filename_base='tmp_parallel_shp',
    tmp_save_path='./tmp/',
    activations_key='activations'
    #use_slurm=False
):
    value_names = ['shpv', 'means']
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
    if os.path.isfile(shpv_path):
        if rewrite_hdf5:
            os.remove(shpv_path)
        else:
            raise RuntimeError(f'File {shpv_path} already exists; rewrite_hdf5={rewrite_hdf5}')
    #########
    with h5py.File(activations_path, 'r') as activ:
        variables_sigmas = activ.attrs['sigmas']
    for module_name in module_names:
        #if not use_slurm or task_id == min_task_id:
        info_msg = f'Processing {module_name}...'
        if verbose:
            print(info_msg, end='\r')
        if log_path is not None:
            logging.info(info_msg)
        ####
        with h5py.File(activations_path, 'r') as activ:
            module_group = activ[module_name]
            activations = module_group[activations_key]
            if Njobs == 1:
                value_arrays = serial_process_shpv(
                    activations,
                    value_names,
                    variables_sigmas,
                    Nvariables,
                    Nouter_samples,
                    Ninner_samples,
                    buffer_size=buffer_size,
                    class_variable=class_variable
                )
            else:
                value_arrays = parallel_process_shpv(
                    activations,
                    value_names,
                    variables_sigmas,
                    Nvariables,
                    Nouter_samples,
                    Ninner_samples,
                    Njobs,
                    buffer_size=buffer_size,
                    filename_base=tmp_filename_base,
                    save_path=tmp_save_path,
                    class_variable=class_variable,
                    #use_slurm=use_slurm
                )
        #if use_slurm and task_id != min_task_id:
        #    continue
        with h5py.File(shpv_path, 'a') as shapley_values_hdf5:
            module_group_hdf5_dataset = shapley_values_hdf5.create_group(module_name)
            for name in value_names:
                module_group_hdf5_dataset.create_dataset(
                    name, data=value_arrays[name], compression="gzip", chunks=True,
                    #maxshape=si_array_shape[:2] + (None, ) + si_array_shape[2:]
                )
    #if use_slurm and task_id != min_task_id:
    #    return ''
    utils.copy_attrs_hdf5(activations_path, shpv_path)
    info_msg = f'Finished.'
    if verbose:
        print(info_msg)
    if log_path is not None:
        logging.info(info_msg)
    return shpv_path

def serial_process_shpv(
    module_activations,
    value_names,
    variables_sigmas,
    Nvariables,
    Nouter_samples,
    Ninner_samples,
    buffer_size=1000,
    class_variable=True
):
    def target_function(a_samples):
        shapley_values = compute_shapley_effect_of_activations(
            a_samples, variables_sigmas, Nvariables, Nouter_samples, Ninner_samples
        )
        mean_activations = np.nanmean(a_samples, axis=0)
        #tmp0 = tmp0.astype(dtype)
        return shapley_values, mean_activations # tuple or list!..
    
    value_arrays = utils.launch_serial_work(
        target_function, module_activations, value_names,
        buffer_size=buffer_size, class_variable=class_variable, #n_verbose=10, activations2=None,
        vectorized_target=True
    )
    return value_arrays

def parallel_process_shpv(
    module_activations,
    value_names,
    variables_sigmas,
    Nvariables,
    Nouter_samples,
    Ninner_samples,
    Njobs=2,
    buffer_size=1000,
    filename_base='tmp_parallel_shpv',
    save_path='./tmp/',
    class_variable=True,
    #use_slurm=False
):
    assert Njobs > 1
    
    def target_function(a_samples):
        shapley_values = compute_shapley_effect_of_activations(
            a_samples, variables_sigmas, Nvariables, Nouter_samples, Ninner_samples
        )
        mean_activations = np.nanmean(a_samples, axis=0)
        #tmp0 = tmp0.astype(dtype)
        return shapley_values, mean_activations # tuple or list!..
    
    value_arrays = utils.launch_parallel_work(
        Njobs, target_function, module_activations, value_names, buffer_size=buffer_size,
        filename_base=filename_base, save_path=save_path, class_variable=class_variable,
        #n_verbose=10, n_trials=10, activations2=None,
        vectorized_target=True, #use_slurm=use_slurm
    )
    return value_arrays

                    
def compute_shapley_effect_of_activations(
    activations, variables_sigmas, Nvariables, Nouter_samples, Ninner_samples
):
    '''
    in progress, not checked yet
    '''
    #Nvariables = problem['Nvariables'] # should be a number of groups of variables !
    Nactivations = activations.shape[-1]
    Npermutations = len(variables_sigmas)
    
    shapley_values = np.zeros([Nvariables, Nactivations])
    
    variables_arange = list(range(Nvariables))
    ind = 0
    #offset = Ninner_samples
    offset = Ninner_samples*Nouter_samples
    
    for i_vperm, variables_sigma_ind in enumerate(variables_sigmas):
        if not isinstance(variables_sigma_ind, (np.ndarray, tuple, list)):
            variables_sigma = custom_compose.permute_by_index(variables_sigma_ind, variables_arange)
        else:
            variables_sigma = variables_sigma_ind
        #sigma = custom_compose.permute_by_index(sigmas[i_perm], transform_variables_arange)
        #local_sigma = copy.copy(sigma)
        #if class_variable_name is not None:
        #    local_sigma.remove(class_variable_ind)
        prev_c_value = 0.
        for i_variable in range(Nvariables):
            # the last one is just a total variation but the permutation
            # matters!
            # conditioned_var_activations
            current_i_var = variables_sigma[i_variable]
            #conditioned_var_activations = np.empty((Nouter_samples, Nactivations))
            #for i_outer in range(Nouter_samples):
            #    activation_values = activations[ind:ind+offset]
            #    conditioned_var_activations[i_outer, :] = np.var(activation_values, ddof=1, axis=0)
            #    ind += offset
            activation_values = activations[ind:ind+offset, :].reshape(
                [Nouter_samples, Ninner_samples, Nactivations]
            )
            ind += offset
            # conditioned_var_activations
            # ignoring NaN's
            #means_inner = np.nanmean(activation_values, axis=1, keepdims=True)
            #activation_values = np.sum(np.power(activation_values - means_inner, 2)/(Ninner_samples-1), axis=1)
            activation_values = np.nanvar(activation_values, ddof=1, axis=1)
            # new_c_value
            # ignoring NaN's
            activation_values = np.nanmean(activation_values, axis=0)
            # delta_c = activation_values - prev_c_value
            shapley_values[current_i_var] += activation_values - prev_c_value # delta_c
            prev_c_value = activation_values
    shapley_values /= Npermutations
    return shapley_values


def shptv_extract_augmented_images(
    config_dict,
    sample_indices,
    traced_parameters_path,
    save_path,
    rewrite=True,
    buffer_size=500,
    labels_attr_name='targets'
):
    hdf5_save_dirname, _ = os.path.split(save_path)
    os.makedirs(hdf5_save_dirname, exist_ok=True)
    if os.path.isfile(save_path):
        if not rewrite:
            raise ValueError
        os.remove(save_path)
    
    traced_parameters = np.load(traced_parameters_path)
    
    subset_size = config_dict['subset_size']
    subset_random_state = config_dict['subset_random_state']
    augmentation_set_number = 2
    dataset_part = config_dict['dataset_part']
    augset1_p = config_dict['augset1_p']
    augset2_p = config_dict['augset2_p']
    use_permutation_variable = config_dict['use_permutation_variable']
    use_partition_variable = config_dict['use_partition_variable']
    problem_sampler_seed = config_dict['problem_sampler_seed']
    class_sampler_seed = config_dict['class_sampler_seed']
    class_selector_seed = config_dict['class_selector_seed']
    permutation_random_seed = config_dict['permutation_random_seed']
    partition_sampler_seed = config_dict['partition_sampler_seed']

    dataset_loader_kwargs_dict = {} # no options
    batch_size = None
    num_workers = 0
    
    image_shape = (3, desired_image_height, desired_image_width)

    dataset_train = None
    dataset_valid = None
    dataset_test = None
    len_dataset = 0
    if 'train' in dataset_part:
        (dataset_train, ), _, _, num_classes = data_loader.general.load_train_valid(
            data_loader.imagenet.CustomImageNet,
            dataset_loader_kwargs_dict,
            data_dirname,
            batch_size,
            num_workers=0,
            augmentation_set_train=None,
            augmentation_set_valid=None,
            train_size=subset_size,
            valid_size=None,
            splitting_random_state=subset_random_state,
            return_torch_dataset=True,
            return_torch_loader=False,
            return_plain_data=False,
            labels_attr_name=labels_attr_name
        )
        len_dataset += 1
    if 'valid' in dataset_part:
        _, (dataset_valid, ), _, num_classes = data_loader.general.load_train_valid(
            data_loader.imagenet.CustomImageNet,
            dataset_loader_kwargs_dict,
            data_dirname,
            batch_size,
            num_workers,
            augmentation_set_train=None,
            augmentation_set_valid=None,
            train_size=None,
            valid_size=subset_size,
            splitting_random_state=subset_random_state,
            return_torch_dataset=True,
            return_torch_loader=False,
            return_plain_data=False,
            labels_attr_name=labels_attr_name
        )
        len_dataset += 1
    if dataset_part == 'test':
        raise NotImplementedError

    dataset = []
    if dataset_train is not None:
        dataset.append(dataset_train)
    if dataset_valid is not None:
        dataset.append(dataset_valid)
    if dataset_test is not None:
        dataset.append(dataset_test)
    
    if len(dataset) == 1:
        dataset = dataset[0]
    
    (
        problem, transform_functions_list, Vi_groups, Vtheta0_dict
    ) = sensitivity_analysis.build_problem_shpv.build_augmented_classification_problem(
        augmentation_set_number,
        image_shape,
        num_classes,
        p_aug_set1=augset1_p,
        p_aug_set2=augset2_p,
        use_permutation_variable=use_permutation_variable,
        use_partition_variable=use_partition_variable
    )
    sampler = sensitivity_analysis.shapley.CustomSampler(
        dataset,
        problem,
        Vi_groups,
        theta_0=Vtheta0_dict,
        random_seed=problem_sampler_seed,
        class_sampler_seed=class_sampler_seed,
        class_selector_seed=class_selector_seed,
        permutation_random_seed=permutation_random_seed,
        partition_sampler_seed=partition_sampler_seed,
        p_test_partition=0.5,
        labels_attr_name=labels_attr_name
    )

    use_class_variable = sampler.class_selector is not None

    if not use_partition_variable:
        current_dataset = dataset

    Nvariables = len(sampler.variables)
    variable_names = list(sampler.variables_names_list)
    
    part_augX, part_origX = [], []
    sigma = None
    
    Nind_samples = len(sample_indices)
    
    for i_sample in range(Nind_samples):
        sampleV = collections.OrderedDict()
        for i, vname in enumerate(variable_names):
            sampleV[i] = df[vname][ind_nan[0][i_sample]]
        ind_var = Nvariables
        if use_partition_variable:
            ind_var -= 1
            current_partition = sampleV.pop(ind_var)[0]
            current_dataset = dataset[current_partition]
        if use_class_variable:
            ind_var -= 1
            indX = sampleV.pop(ind_var)[0]#.astype('i')[0]
            sampleX, _ = current_dataset[indX]
        else:
            raise NotImplementedError

        if use_permutation_variable:
            ind_var -= 1
            sigma = sampleV.pop(ind_var)[0].tolist()

        if pre_processing_functions is not None:
            sampleX = pre_processing_functions(sampleX)
        partX0.append(sampleX)
        sampleX = sensitivity_analysis.shapley.composed_transform(
            sampleX,
            transform_functions_list,
            sampleV,
            sampler.theta_indices,
            sampler.use_switch_variables_indices,
            sigma
        )
        if post_processing_functions is not None:
            sampleX = post_processing_functions(sampleX)
            partX0[-1] = post_processing_functions(partX0[-1])
        partX.append(sampleX)
        if ((i_sample+1) % buffer_size == 0) or (i_sample+1 == Nind_samples):
            partX = torch.cat(partX).detach().numpy()
            partX0 = torch.cat(partX0).detach().numpy()
            data_dict = {'original': partX0, 'augmented': partX}
            utils.update_data_only_hdf5(data_dict, hdf5_file_path)
            del partX, partX0
            partX, partX0 = [], []        
    return hdf5_file_path

def normalize_shapley(
    shp_values, tv_estimates=None, eps=1e-8, zero_atol=1e-8, carefully=True, clip_negative=True,
    #debug=True
):
    if tv_estimates is None:
        if clip_negative:
            tv_estimates = np.clip(shp_values, 0., None).sum(axis=0, keepdims=True)
        else:
            tv_estimates = shp_values.sum(axis=0, keepdims=True)
    if not carefully:
        if clip_negative:
            return np.clip(shp_values, 0, None) / (eps+tv_estimates)
        else:
            return shp_values / (eps+tv_estimates)
    tv_estimates = tv_estimates[0]
    #ind0 = np.where(np.isclose(tv_estimates, 0., atol=zero_atol))
    #ind = np.where(~np.isclose(tv_estimates, 0., atol=zero_atol))
    
    rv = shp_values.copy()
    #if clip_negative:
    #    rv[ind] = np.clip(rv[ind], 0., None)
    #rv[ind] /= eps+tv_estimates[ind]
    #rv[ind0] = 0.
    for i in range(len(rv)):
        if clip_negative:
            #rv[i][ind] = np.clip(rv[i][ind], 0., None)
            rv[i] = np.clip(rv[i], 0., None)
        #ind0 = np.where(np.isclose(tv_estimates, 0., atol=zero_atol))
        #ind = np.where(~np.isclose(tv_estimates, 0., atol=zero_atol))
        ind0 = np.where(np.isclose(rv[i], 0., atol=zero_atol))
        ind = np.where(~np.isclose(rv[i], 0., atol=zero_atol))
        rv[i][ind] /= eps+tv_estimates[ind]
        rv[i][ind0] = 0.
    #if debug:
    #    tmp = np.sum(rv, axis=0)
    #    print(tmp.min(), tmp.max())
    return rv








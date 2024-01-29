#import sys
#sys.path.append('../src')
import os
import math
import copy
#import multiprocessing
import concurrent.futures

import logging
import collections
import shutil

import h5py
import numpy as np

import custom_compose

from . import contrast
from data_loader.general import get_dataset_attr
from hdf5_utils import copy_attrs_hdf5, save_activations, update_data_only_hdf5


_class_variable_name = 'class'
_permutation_variable_name = 'permutation'
_partition_variable_name = 'partition'


def split_seed(seed, n_splits):
    rng = np.random.default_rng(seed)
    return rng.integers(0, int(2**16), size=(n_splits,))

def custom_merge_dicts(dict_in, dict_from):
    for key in dict_from:
        dict_in[key] = dict_in.get(key, [])
        dict_in[key].append(dict_from[key])
    return dict_in

class ClassRandomSelector(object):
    
    def __init__(
        self,
        num_classes,
        stratified=True,
        random_seed=0,
        permute_result=False
    ):
        self.num_classes = num_classes
        self.stratified = stratified
        if self.stratified:
            self.candidates = np.ones(self.num_classes, dtype='i')
        self.permute_result = permute_result
        self.seed(random_seed)
        
    def seed(self, random_seed):
        if not self.permute_result:
            self.rng = np.random.default_rng(random_seed)
            self.permutation_rng = None
            return
        if isinstance(random_seed, int):
            random_seed = split_seed(random_seed, 2)
        self.rng = np.random.default_rng(random_seed[0])
        self.permutation_rng = np.random.default_rng(random_seed[1])
    
    def sample(self, N=1):
        if not self.stratified:
            return self.rng.integers(0, self.num_classes, size=N)
        
        #N_not_sampled = (self.candidates > 0).sum()
        
        s = np.where(self.candidates > 0)[0]
        s = self.rng.permutation(s)
        len_s = len(s)
        if len_s == N: # <= num_classes
            self.candidates.fill(1)
            #return self.rng.permutation(s)
            if self.permute_result:
                return self.permute_rng.permutation(s)
            return s
        if len_s > N: # <= num_classes
            if self.permute_result:
                s = self.permutation_rng.permutation(s)[:N]
            else:
                s = s[:N]
            self.candidates[s] = 0
            return s
        # len_s < N 
        self.candidates[s] = 0
        Ns = N - len_s # > 0
        a = np.arange(self.num_classes, dtype='i')
        #a = self.rng.permutation(a)
        if Ns > self.num_classes:
            m = int(np.ceil(Ns/self.num_classes)) # >= 2
            b = np.kron(a, np.ones(m-1, dtype='i'))
            a = np.append(b, a)
        s = np.append(s, a[:Ns])
        if Ns == len(a):
            self.candidates.fill(1)
        else:
            self.candidates[a[Ns:]] = 1
        if self.permute_result:
            return self.permute_rng.permutation(s)
        return s
    
    def __call__(self, N=1):
        return self.sample(N)

class PartitionSampler(object):
    def __init__(
        self,
        n_partitions,
        p=None,
        random_seed=0
    ):
        self.n_partitions = n_partitions
        self.seed(random_seed)
        self.p = p
        if p is not None:
            if isinstance(p, float):
                assert self.n_partitions == 2
                self.p = [1-p, p]
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
        
    def sample(self, N=1):
        if self.p is None:
            return self.rng.integers(min=0, max=self.n_partitions, size=N)
        return self.rng.choice(a=self.n_partitions, size=N, replace=True, p=self.p)
    

class OneClassSampler(object):#torch.utils.data.sampler.Sampler):
    def __init__(self, y, random_seed=0):
        self.num_samples = len(y) 
        self.classes, self.inv_ind, self.num_samples_per_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        self.num_classes = len(self.classes)
        self.already_sampled_indices = [set() for i in range(self.num_classes)]
        
        self.seed(random_seed)
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
    
    def sample(self, class_ind):
        if len(self.already_sampled_indices[class_ind]) == self.num_samples_per_class[class_ind]:
            self.already_sampled_indices[class_ind] = set()
        candidates = list(
            set(np.where(self.inv_ind == self.classes[class_ind])[0]).difference(
                self.already_sampled_indices[class_ind]
            )
        )
        ind = self.rng.choice(candidates)
        self.already_sampled_indices[class_ind].add(ind)
        return ind
    
class PermutationRandomSampler(object):
    def __init__(
        self,
        n_variables,
        oversample=False,
        permute_result=True,
        permutation_seed=0
    ):
        self.n_variables = n_variables
        
        self.permute_result = permute_result
        self.oversample = oversample
        
        self.max_num_sigmas = math.factorial(n_variables)
        self.variables_arange = list(range(n_variables))
        
        self.seed(permutation_seed)
    
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
        if self.permute_result:
            self.rng_permute = np.random.default_rng(random_seed)
    
    def sample(self, N, return_number=False):
        
        sigmas = self.rng.choice(
            self.max_num_sigmas, size=min(N, self.max_num_sigmas), replace=False, shuffle=True
        ) # replace=False is essential here!
        n_samples = len(sigmas)
        add_more_samples = self.oversample and (n_samples < N)
        Nmax = N - n_samples
        while add_more_samples:
            new_sigmas = self.rng.choice(
                self.max_num_sigmas, size=min(Nmax, self.max_num_sigmas), replace=False, shuffle=True
            )
            sigmas = np.append(sigmas, new_sigmas, axis=-1)
            n_samples = len(sigmas)
            assert n_samples <= N
            Nmax = N - n_samples
            add_more_samples = (n_samples < N)
        if self.permute_result:
            sigmas = self.rng_permute.permutation(sigmas)
        if not return_number:
            sigmas = list(
                map(lambda ind: custom_compose.permute_by_index(ind, self.variables_arange), sigmas)
            )
        return sigmas
    

def convert_uniform_to_int(uniform_value, max_int):
    assert np.all(uniform_value >= 0) and np.all(uniform_value < 1)
    return np.floor(uniform_value*max_int).astype('i')

def get_ordict_key_index(ordict, key):
    assert isinstance(ordict, collections.OrderedDict)
    ind = list(ordict.keys()).index(key)
    return ind

def extract_group_boundary_indices(group_array, num_vars=None):
    if num_vars is None:
        num_vars = len(group_array)
    sorted_names, index = np.unique(group_array, return_index=True)
    argind = np.argsort(index)
    sorted_names, index = sorted_names[argind], index[argind]
    index = np.append(index, num_vars)
    return index

def worker(
    i_exec,
    i_job,
    target_function,
    X,
    X2,
    value_names,
    save_path,
    filename_base,
    n_verbose=10,
    #Njobs=None,
    vectorized_target=False
):
    #if Njobs is not None:
    #    barrier = multiprocessing.Barrier(Njobs, timeout=None)
    n_values = len(value_names) # list / tuple / ...
    assert n_values > 0 
    Noutputs = X.shape[-1]
    #value_arrays = collections.OrderedDict((name, []) for name in value_names)
    if vectorized_target:
        if X2 is None:
            tmp = target_function(X)
        else:
            tmp = target_function(X, X2) 
        value_arrays = dict(zip(value_names, tmp))
        info_msg = f'i_exec={i_exec}, i_job={i_job}: pworker finished'
        logging.info(info_msg)
    else:
        value_arrays = dict((name, []) for name in value_names)
        for i_out in range(Noutputs):
            #print('enter target')
            if X2 is None:
                tmp = target_function(X[:, i_out])
            else:
                tmp = target_function(X[:, i_out], X2[:, i_out])
            for i, name in enumerate(value_names):
                if tmp[i] is not None:
                    value_arrays[name].append(tmp[i].T)
            #print('exit target')
            if (i_out+1) % n_verbose == 0:
                info_msg = f'i_exec={i_exec}, i_job={i_job}: pworker, {i_out+1}/{Noutputs}'
                logging.info(info_msg)
        for name in value_names:
            if len(value_arrays[name]) == 0:
                continue
            value_arrays[name] = np.array(value_arrays[name]).T
    if save_path is None:
        return value_arrays
    with h5py.File(
        os.path.join(save_path, f'i_exec={i_exec}_i_job={i_job}_{filename_base}.hdf5'), 'w'
    ) as local_file:
        for name in value_names:
            if len(value_arrays[name]) == 0:
                continue
            local_file.create_dataset(
                name, data=value_arrays[name]#, compression="gzip", chunks=True,
                #maxshape=si_array_shape[:2] + (None, ) + si_array_shape[2:]
            )
    #if Njobs is not None:
    #    barrier.wait()
    return

def launch_parallel_work(
    Njobs,
    target_function,
    activations,
    value_names,
    buffer_size=1000,
    filename_base='tmp_parallel',
    save_path='./tmp/',
    class_variable=False,
    n_verbose=10,
    n_trials=10,
    activations2=None,
    vectorized_target=False,
    #use_slurm=False
):
    #if use_slurm:
    #    # https://slurm.schedmd.com/job_array.html
    #    # os.environ["SLURM_JOB_ID"] # <- should be taken into account if there are concurrent jobs
    #
    #    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    #    min_task_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
    #    max_task_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
    current_shape = activations.shape
    os.makedirs(save_path, exist_ok=True)
    if activations2 is None:
        cur_activations2 = None
    
    if not class_variable:
        Nbatch = current_shape[1]
        Noutputs = int(np.prod(current_shape[2:]))
    else:
        Noutputs = int(np.prod(current_shape[1:]))
    Njobs = min(Noutputs, Njobs)
    #if use_slurm:
    #    Njobs = min(Njobs, max_task_id-min_task_id+1)
    Nout_per_job = Noutputs // Njobs 
    Num_hardworkers = Noutputs % Njobs
    
    Nexec = int(np.ceil(Nout_per_job / buffer_size))
    buffer_size = min(buffer_size, Nout_per_job)
    ind = 0
    for i_exec in range(Nexec):
        #offset = (ind+1-Noutputs) // Njobs
        
        offset = min((Noutputs-ind+1)//Njobs, buffer_size)
        ind_copy = ind
        
        filenames = os.listdir(save_path)
        filenames = list(
            filter(lambda x: x.startswith(f'i_exec={i_exec}_i_job='), filenames)
        )
        
        #i_trial = 0
        #while (i_trial < n_trials) and (len(filenames) < Njobs):
        #    #print('inside while')
        #    job_list = []
        #    i_trial += 1
        #    ind = ind_copy
        #    for i_job in range(Njobs):
        #        cur_offset = offset
        #        if (i_exec == Nexec-1) and (i_job < Num_hardworkers):
        #            cur_offset += 1
        #        if f'i_exec={i_exec}_i_job={i_job}_{filename_base}.hdf5' in filenames:
        #            ind += cur_offset
        #            continue
        #        #if use_slurm and (i_job != (task_id-min_task_id)):
        #        #    ind += cur_offset
        #        #    continue
        #        #print(activations.shape, ind, ind+cur_offset)
        #        if class_variable:
        #            cur_activations = activations[:, ind:ind+cur_offset]
        #            if activations2 is not None:
        #                cur_activations2 = activations2[:, ind:ind+cur_offset]
        #        else:
        #            cur_activations = activations[:, :, ind:ind+cur_offset]
        #            cur_activations = np.reshape(
        #                cur_activations, [current_shape[0]*Nbatch, -1]#current_shape[1], -1]
        #            )
        #            if activations2 is not None:
        #                cur_activations2 = activations2[:, :, ind:ind+cur_offset]
        #                cur_activations2 = np.reshape(
        #                    cur_activations2, [-1, cur_activations.shape[-1]]#current_shape[1], -1]
        #                )
        #        #if use_slurm:
        #        #    worker(
        #        #        i_exec, i_job, target_function, cur_activations, cur_activations2,
        #        #        value_names, save_path, filename_base, n_verbose, #Njobs,
        #        #        vectorized_target
        #        #    )
        #        #else:
        #        ##cur_proc = multiprocessing.Process(
        #        ##    target=worker, args=(
        #        ##        i_exec, i_job, target_function, cur_activations, cur_activations2,
        #        ##        value_names, save_path, filename_base, n_verbose, #Njobs,
        #        ##        vectorized_target
        #        ##    )
        #        ##)
        #        ###########
        #    
            #ind = ind_copy
        indices = []
        for i_job in range(Njobs):
            cur_offset = offset
            if (i_exec == Nexec-1) and (i_job < Num_hardworkers):
                cur_offset += 1
            if f'i_exec={i_exec}_i_job={i_job}_{filename_base}.hdf5' in filenames:
                ind += cur_offset
                indices.append(None)
                continue
            indices.append((ind, ind+cur_offset))
            ind += cur_offset
        # https://www.digitalocean.com/community/tutorials/how-to-use-threadpoolexecutor-in-python-3-ru
        #with concurrent.futures.ProcessPoolExecutor(max_workers=Njobs) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=Njobs) as executor:
            futures_list = []
            Nout_actual_list = []
            for i_job in range(Njobs):
                if indices[i_job] is None:
                    Nout_actual_list.append(None)
                    continue
                ind0, ind1 = indices[i_job]
                if class_variable:
                    cur_activations = activations[:, ind0:ind1]
                    if activations2 is not None:
                        cur_activations2 = activations2[:, ind0:ind1]
                else:
                    cur_activations = activations[:, :, ind0:ind1]
                    cur_activations = np.reshape(
                        cur_activations, [current_shape[0]*Nbatch, -1]#current_shape[1], -1]
                    )
                    if activations2 is not None:
                        cur_activations2 = activations2[:, :, ind0:ind1]
                        cur_activations2 = np.reshape(
                            cur_activations2, [-1, cur_activations.shape[-1]]#current_shape[1], -1]
                        )
                Nout_actual_list.append(cur_activations.shape[-1])
                
                futures_list.append(
                    executor.submit(
                        worker,
                        i_exec, i_job, target_function, cur_activations, cur_activations2,
                        value_names, save_path, filename_base, n_verbose, #Njobs,
                        vectorized_target
                    )
                )
                info_msg = (
                    f"{i_exec+1}/{Nexec}: process {i_job} submitted "
                    f" [{ind0}--{ind1}]"
                )
                logging.info(info_msg)
            #concurrent.futures.wait(futures_list)
            for i_job, future_instance in enumerate(concurrent.futures.as_completed(futures_list)):
                # we have a strong prior on the order because i/o operations take a lot of time
                if indices[i_job] is None:
                    continue
                ind0, ind1 = indices[i_job]
                Nout_actual = Nout_actual_list[i_job]
                #if not use_slurm:
                ##job_list.append(cur_proc)
                ##cur_proc.start()
                ###
                info_msg = (
                    f"{i_exec+1}/{Nexec}: process {i_job} with {Nout_actual} samples"
                    f" [{ind0}--{ind1}]"
                )
                logging.info(info_msg)
                #ind += cur_offset
            #if not use_slurm:
            # block until ready
            ## for p in job_list:
            ##    #if p.is_alive():
            ##    p.join()
            ##for p in job_list:
            ##    try:
            ##       p.close()
            ##    except:
            ##        print('p.close() exception')
            #######
            ##filenames = os.listdir(save_path)
            ##filenames = list(
            ##    filter(lambda x: x.startswith(f'i_exec={i_exec}_i_job='), filenames)
            ##)
    
    #if not use_slurm or (task_id == min_task_id):
    info_msg = "========================================================"
    logging.info(info_msg)
    info_msg = "Launcher has successfully finished his work"
    logging.info(info_msg)
    value_arrays = gather_results(
        Nexec, Njobs, save_path, filename_base, value_names
    )
    return value_arrays
    #return None
    
def gather_results(
    Nexec,
    Njobs,
    save_path,
    filename_base,
    value_names,
):
    #si_array, si2_array = None, None
    value_arrays = dict((name, None) for name in value_names)
    local_paths = []
    for i_exec in range(Nexec):
        job_list = []
        for i_job in range(Njobs):
            local_path = os.path.join(
                save_path, f'i_exec={i_exec}_i_job={i_job}_{filename_base}.hdf5'
            )
            local_paths.append(local_path)
            with h5py.File(local_path, 'r') as local_file:
                for name in value_names:
                    if value_arrays[name] is None:
                        value_arrays[name] = local_file[name][:]
                    else:
                        #print(i_exec, i_job, name, value_arrays[name].shape, local_file[name].shape)
                        value_arrays[name] = np.append(value_arrays[name], local_file[name][:], axis=-1)
    for local_path in local_paths:
        os.remove(local_path)
    return value_arrays

def launch_serial_work(
    target_function,
    activations,
    value_names,
    buffer_size=1000,
    class_variable=False,
    n_verbose=10,
    activations2=None,
    vectorized_target=False
):
    current_shape = activations.shape
    if activations2 is None:
        cur_activations2 = None
    
    if not class_variable:
        Nbatch = current_shape[1]
        Noutputs = int(np.prod(current_shape[2:]))
    else:
        Noutputs = int(np.prod(current_shape[1:]))
    
    Nact_blocks = 1
    if buffer_size is not None:
        Nact_blocks = int(np.ceil(Noutputs/buffer_size))
    
    value_arrays = None
    ind = 0
    offset = buffer_size or Noutputs # Noutputs if buffer_size is None
    for k in range(Nact_blocks):
        #print(activations.shape, ind, ind+cur_offset)
        if class_variable:
            cur_activations = activations[:, ind:ind+offset]
            if activations2 is not None:
                cur_activations2 = activations2[:, ind:ind+offset]
        else:
            cur_activations = activations[:, :, ind:ind+offset]
            cur_activations = np.reshape(
                cur_activations, [current_shape[0]*Nbatch, -1]#current_shape[1], -1]
            )
            if activations2 is not None:
                cur_activations2 = activations2[:, :, ind:ind+offset]
                cur_activations2 = np.reshape(
                    cur_activations2, [-1, cur_activations.shape[-1]]#current_shape[1], -1]
                )
            
        cur_value_arrays = worker(
            i_exec=0, i_job=0, target_function=target_function,
            X=cur_activations, X2=cur_activations2, value_names=value_names,
            save_path=None, filename_base=None, n_verbose=10, #Njobs=None,
            vectorized_target=vectorized_target
        )
        if value_arrays is None:
            value_arrays = cur_value_arrays
        else:
            for name in value_names:
                ######## ERROR: np.append() flattens the n-dimensional input!!
                value_arrays[name] = np.append(
                    value_arrays[name],
                    cur_value_arrays[name],
                    axis=-1 # !!!!!!!!!! prevents flattening
                )
                
        Nout_actual = cur_activations.shape[-1]
        info_msg = (
            f"X/X: serial process with {Nout_actual} samples"
            f" [{ind}--{ind+offset}]"
        )
        logging.info(info_msg)
        ind += offset

    info_msg = "========================================================"
    logging.info(info_msg)
    info_msg = "Launcher has successfully finished his work"
    logging.info(info_msg)
    return value_arrays


def extract_switch_variables_indices(problem):
    exclude_list = [
        _class_variable_name,
        _permutation_variable_name,
        _partition_variable_name
    ]

    groups = problem['groups']
    names = problem['names']
    inds = []
    gind = 0
    n_groups = len(set(groups))
    for i_g in range(n_groups):
        current_group = groups[gind]
        while (gind < len(groups)) and (groups[gind] == current_group):
            gind += 1
        if (current_group in names) and (not current_group in exclude_list):
            inds.append(i_g)
    return inds
                
def add_suffix_to_path(path, suffix):
    root, ext = os.path.splitext(path)
    return f'{root}_{suffix}.{ext}'

def correct_pvalues_within_hdf5(
    cs_path,
    module_names,
    values_name='cs_pvals',
    alpha=0.95,
    method='fdr_by',
    suffix='fdr-by-corrected'
):
    new_cs_path = add_suffix_to_path(cs_path, 'fdr-by-corrected')
    # save a copy of each hdf5 file
    shutil.copy2(cs_path, new_cs_path);
    # extract pvalues dict
    with h5py.File(new_cs_path, 'r') as cs:
        pvalues_dict = {}
        for i_mn, module_name in enumerate(module_names):
            pvalues_dict[module_name] = cs[f'{module_name}/{values_name}'][:]
    # correct it for multiple testing
    corrected_pvalues_dict = contrast.multiple_tests_corection(
        pvalues_dict, alpha=alpha, method='fdr_by'
    )
    # reassign corrected values
    with h5py.File(new_cs_path, 'a') as cs:
        for i_mn, module_name in enumerate(module_names):
            cs[f'{module_name}/{values_name}'][...] = pvalues_dict[module_name]
    return new_cs_path
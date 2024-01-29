import logging
import h5py
import os

import numpy as np

import torch_learning
import custom_compose
from . import utils


def compute_total_variances(
    #model,
    module_names,
    activations_path,
    log_path='glance_at_tv.log',
    tv_path='total_variances.hdf5',
    Njobs=2,
    buffer_size=1000,
    class_variable=True,
    verbose=True,
    rewrite_hdf5=False,
    tmp_filename_base='tmp_parallel_tv',
    tmp_save_path='./tmp/',
    activations_key='activations'
    #use_slurm=False
):
    value_names = ['mean', 'var']
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
    if os.path.isfile(tv_path):
        if rewrite_hdf5:
            os.remove(tv_path)
        else:
            raise RuntimeError(f'File {tv_path} already exists; rewrite_hdf5={rewrite_hdf5}')
    ##
    
    for module_name in module_names:
        #if not use_slurm or task_id == min_task_id:
        info_msg = f'Processing {module_name}...'
        if verbose:
            print(info_msg, end='\r')
        if log_path is not None:
            logging.info(info_msg)
        ##
        with h5py.File(activations_path, 'r') as activ:
            module_group = activ[module_name]
            activations = module_group[activations_key]
            if Njobs == 1:
                value_arrays = serial_process_tv(
                    activations,
                    value_names,
                    buffer_size=buffer_size,
                    class_variable=class_variable
                )
            else:
                value_arrays = parallel_process_tv(
                    activations,
                    value_names,
                    Njobs,
                    buffer_size=buffer_size,
                    filename_base=tmp_filename_base,
                    save_path=tmp_save_path,
                    class_variable=class_variable,
                    #use_slurm=use_slurm
                )
        #if use_slurm and task_id != min_task_id:
        #    continue
        with h5py.File(tv_path, 'a') as total_variances_hdf5:
            module_group_hdf5_dataset = total_variances_hdf5.create_group(module_name)
            for name in value_names:
                module_group_hdf5_dataset.create_dataset(
                    name, data=value_arrays[name], compression="gzip", chunks=True,
                    #maxshape=si_array_shape[:2] + (None, ) + si_array_shape[2:]
                )
    #if use_slurm and task_id != min_task_id:
    #    return ''
    utils.copy_attrs_hdf5(activations_path, tv_path)
    info_msg = f'Finished.'
    if verbose:
        print(info_msg)
    if log_path is not None:
        logging.info(info_msg)
    return tv_path

def serial_process_tv(
    module_activations,
    value_names,
    buffer_size=1000,
    class_variable=False
):
    value_arrays = utils.launch_serial_work(
        compute_total_variance_of_activations,
        module_activations,
        value_names,
        buffer_size=buffer_size,
        class_variable=class_variable, #n_verbose=10, activations2=None,
        vectorized_target=True
    )
    return value_arrays

def parallel_process_tv(
    module_activations,
    value_names,
    Njobs=2,
    buffer_size=1000,
    filename_base='tmp_parallel_tv',
    save_path='./tmp/',
    class_variable=True,
    #use_slurm=False
):
    assert Njobs > 1
    value_arrays = utils.launch_parallel_work(
        Njobs,
        compute_total_variance_of_activations,
        module_activations,
        value_names,
        buffer_size=buffer_size,
        filename_base=filename_base,
        save_path=save_path,
        class_variable=class_variable, #n_verbose=10, n_trials=10, activations2=None,
        vectorized_target=True,
        #use_slurm=use_slurm
    )
    return value_arrays

def compute_total_variance_of_activations(activation_values):
    ddof = 1
    mean_activations = np.nanmean(activation_values, axis=0)
    #var_activations = np.sum((activation_values - mean_activations)**2)/(N_V-1)
    var_activations = np.nanvar(activation_values, ddof=ddof, axis=0)
    
    # detected ~700-800 cases of 262k samples
    return [mean_activations, var_activations]
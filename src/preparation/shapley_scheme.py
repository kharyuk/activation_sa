import copy

import h5py
import torch

import numpy as np

import sensitivity_analysis.utils
import sensitivity_analysis.build_problem_shpv
import sensitivity_analysis.shapley

import data_loader.general

from . import hdf5_buffer


class ShapleyFullSampler(object):
    def __init__(
        self,
        dataset,
        problem,
        Vi_groups,
        theta_0,
        Nouter_samples,
        Ninner_samples,
        activations_path,
        variables_permutation_seed=None,
        problem_sampler_seed=None,
        class_sampler_seed=None,
        class_selector_seed=None,
        permutation_sampler_seed=None,
        partition_sampler_seed=None,
        p_test_partition=0.5,
        labels_attr_name='targets'
    ):
        self.problem = problem
        self.Nvariables = problem['num_vars']
        self.Nouter_samples = Nouter_samples
        self.Ninner_samples = Ninner_samples
        self.variables_arange = list(range(self.Nvariables))
        self.variables_permutation_sampler = sensitivity_analysis.utils.PermutationRandomSampler(
            self.Nvariables,
            oversample=True,
            permute_result=False,
            permutation_seed=variables_permutation_seed
        )
        self.activations_path = activations_path
        self.variables_sampler = sensitivity_analysis.shapley.CustomSampler(
            dataset,
            problem,
            Vi_groups,
            theta_0=theta_0,
            random_seed=problem_sampler_seed,
            class_sampler_seed=class_sampler_seed,
            class_selector_seed=class_selector_seed,
            permutation_random_seed=permutation_sampler_seed,
            partition_sampler_seed=partition_sampler_seed,
            p_test_partition=p_test_partition,
            labels_attr_name=labels_attr_name
        )
        
    def __iter__(self):
        local_sigma = None
        indX = None
        current_partition = None
        while True:
            variables_sigma = self.variables_permutation_sampler.sample(N=1, return_number=False)[0]
            if self.activations_path is not None:
                with h5py.File(self.activations_path, 'a') as activ:
                    hdf5_buffer.fill_sigmas(activ, variables_sigma)
            for i_variable in range(self.Nvariables):
                for i_outer in range(self.Nouter_samples):
                    sampleV_right = self.variables_sampler.sample(variables_sigma[i_variable+1:])
                    for i_inner in range(self.Ninner_samples):
                        sampleV = copy.copy(sampleV_right)
                        sampleV = self.variables_sampler.sample(
                            variables_sigma[:i_variable+1], result=sampleV
                        )
                        ind = self.Nvariables
                        if self.variables_sampler.use_partition_variable:
                            ind -= 1
                            current_partition = sampleV.pop(ind)[0]
                        if self.variables_sampler.use_class_variable:
                            ind -= 1
                            indX = sampleV.pop(ind)[0]           
                        else:
                            raise NotImplementedError
                        if self.variables_sampler.use_permutation_variable:
                            ind -= 1
                            local_sigma = sampleV.pop(ind)[0]
                        yield sampleV, local_sigma, indX, current_partition
                        
    def compute_shapley_values(
        self, 
        module_names,        
        shpv_path='shapley_values.hdf5',
        log_path='glance_at_shpv.log', #####
        Njobs=2,
        buffer_size=1000,
        verbose=True,
        rewrite_hdf5=False,
        tmp_filename_base='tmp_parallel_shp',
        tmp_save_path='./tmp/'
    ):
        # variables_sigmas = 
        variables_path = sensitivity_analysis.shapley.analyze_transform_with_shapley(
            module_names,
            self.activations_path,
            ##variables_sigmas,
            self.Nvariables,
            self.Nouter_samples,
            self.Ninner_samples,
            log_path=log_path,
            shpv_path=shpv_path,
            Njobs=Njobs,
            buffer_size=buffer_size,
            class_variable=class_variable,
            verbose=verbose,
            rewrite_hdf5=rewrite_hdf5,
            tmp_filename_base=tmp_filename_base,
            tmp_save_path=tmp_save_path
        )
        return variable_path
                        
                        
                        
class CustomDatasetShapley(object):
    def __init__(
        self,
        dataset_loader_func,
        dataset_loader_func_params_dict,
        data_dirname,
        augmentation_set_number,
        image_shape,
        Nouter_samples,
        Ninner_samples,
        activations_path,
        batch_size=None,
        train_size=0.5,
        valid_size=None,            
        n_samples_per_class=None,
        n_samples_all=None,
        split_classwise=True,
        pre_processing_functions=None,
        use_permutation_variable=True,
        use_partition_variable=True,
        variables_permutation_seed=None,
        problem_sampler_seed=None,
        class_sampler_seed=None,
        class_selector_seed=None,
        permutation_sampler_seed=None,
        partition_sampler_seed=None,
        splitting_random_seed=None,
        equating_random_state=None,
        p_test_partition=0.5,
        labels_attr_name='targets',
        dataset_init_split='train',
        augset1_p=0.5,
        augset2_p=0.5,
    ):
        assert use_permutation_variable != (permutation_sampler_seed is None)
        assert use_partition_variable != (partition_sampler_seed is None)
        
        self.use_permutation_variable = use_permutation_variable
        self.use_partition_variable = use_partition_variable
        
        self.pre_processing_functions = pre_processing_functions
        
        
        # 1. split dataset
        if use_partition_variable:
            assert partition_sampler_seed is not None
            # splitting it into train and valid parts
            (partition1_dataset, ), (partition2_dataset, ), num_classes = (
                data_loader.general.load_single_part_bipartitioned_dataset(
                    dataset_loader_func,
                    dataset_loader_func_params_dict,
                    data_dirname,
                    batch_size=batch_size,
                    num_workers=0,
                    augmentation_set_partition1=None,
                    augmentation_set_partition2=None,
                    n_samples_per_class=n_samples_per_class,
                    n_samples_all=n_samples_all,
                    train_size=train_size,
                    valid_size=valid_size,
                    split_classwise=split_classwise,
                    equating_random_state=equating_random_state,
                    splitting_random_state=splitting_random_seed,
                    return_torch_dataset=True,
                    return_torch_loader=False,
                    return_plain_data=False,
                    labels_attr_name=labels_attr_name,
                    dataset_init_split=dataset_init_split, # 'val'
                    do_pretensorize=False,
                    do_normalize=False,
                    shuffle=False,
                )
            )
            self.dataset = (partition1_dataset, partition2_dataset)
        else:
            partition_sampler_seed = None
            (self.dataset, ), num_classes = data_loader.general.load_single_dataset_part(
                dataset_loader_func,
                dataset_loader_func_params_dict,
                data_dirname,
                batch_size=batch_size,
                num_workers=0,
                augmentation_set=None,
                subset_size=n_samples_all,
                samples_per_class=n_samples_per_class,
                shuffle=False,
                splitting_random_state=splitting_random_state,
                return_torch_dataset=True,
                return_torch_loader=False,
                return_plain_data=False,
                labels_attr_name=labels_attr_name,
                dataset_init_split=dataset_init_split,
                do_pretensorize=False,
                do_normalize=False
            )
        # 2. get problem, functions
        (
            problem, self.transform_functions_list, Vi_groups, Vtheta0_dict
        ) = sensitivity_analysis.build_problem_shpv.build_augmented_classification_problem(
            augmentation_set_number,
            image_shape,
            num_classes,
            p_aug_set1=augset1_p,
            p_aug_set2=augset2_p,
            use_permutation_variable=use_permutation_variable,
            use_partition_variable=use_partition_variable
        )
        # 3. Shapley sampler
        self.Nvariables = problem['num_vars']
        self.Nouter_samples = Nouter_samples
        self.Ninner_samples = Ninner_samples
        
        self.sampler = ShapleyFullSampler(
            self.dataset,
            problem,
            Vi_groups,
            Vtheta0_dict,
            Nouter_samples,
            Ninner_samples,
            activations_path,
            variables_permutation_seed=variables_permutation_seed,
            problem_sampler_seed=problem_sampler_seed,
            class_sampler_seed=class_sampler_seed,
            class_selector_seed=class_selector_seed,
            permutation_sampler_seed=permutation_sampler_seed,
            partition_sampler_seed=partition_sampler_seed,
            p_test_partition=p_test_partition,
            labels_attr_name=labels_attr_name
        )
    
    def __iter__(self, ):
        for sample in self.sampler:
            sampleV, local_sigma, indX, current_partition = sample
            if current_partition is None:
                sampleX, sampleY = self.dataset[indX]
            else:
                sampleX, sampleY = self.dataset[current_partition][indX]
            if self.pre_processing_functions is not None:
                sampleX = self.pre_processing_functions(sampleX)
            sampleX = sensitivity_analysis.shapley.composed_transform(
                sampleX,
                self.transform_functions_list,
                sampleV,
                self.sampler.variables_sampler.theta_indices,
                self.sampler.variables_sampler.use_switch_variables_indices,
                local_sigma
            )
            yield sampleX, sampleY, current_partition
    
    def extract_variables_and_group_indices(self):
        margin = 0
        ind = self.sampler.variables_sampler.theta_indices
        
        if self.sampler.variables_sampler.use_class_variable:
            #ind.remove(self.sampler.variables_sampler.class_variable_ind)
            margin += 1
        if self.sampler.variables_sampler.use_partition_variable:
            #ind.remove(self.sampler.variables_sampler.partition_variable_ind)
            margin += 1
        if self.sampler.variables_sampler.use_permutation_variable:
            #ind.remove(self.sampler.variables_sampler.permutation_variable_ind)
            margin += 1
        
        ind = np.array(ind[:-margin])
        vind = np.arange(len(ind)-1)
        return vind, ind
        
        
    def extract_batch(self, batch_size):
        #train_partX, train_partY = [], []
        #valid_partX, valid_partY = [], []
        partX, partY = [], []
        c_part = []
        i_sample_within_batch = 0
        for (sampleX, sampleY, current_partition) in self:
            partX.append(sampleX)
            partY.append(sampleY)
            if current_partition is None:
                i_sample_within_batch += 1
            else:
                c_part.append(current_partition)
                if current_partition == 0:
                    i_sample_within_batch += 1
            if i_sample_within_batch == batch_size:
                break
        if len(c_part) == 0:
            c_part = None
        else:
            c_part = np.array(c_part)
        partX = torch.stack(partX)
        #partY = torch.cat(partY)
        partY = torch.LongTensor(partY)
        return partX, partY, c_part
            
    def extract_batch_old(self, batch_size):
        #train_partX, train_partY = [], []
        #valid_partX, valid_partY = [], []
        partX, partY = None, None
        i_sample_within_batch = 0
        for (sampleX, sampleY, current_partition) in self:
            if current_partition is None:
                if partX is None:
                    partX, partY = [], []
                partX.append(sampleX)
                partY.append(sampleY)
                i_sample_within_batch += 1
            else:
                if partX is None:
                    partX, partY = {}, {}
                if partX.get(current_partition, None) is None:
                    partX[current_partition] = []
                    partY[current_partition] = []
                partX[current_partition].append(sampleX)
                partY[current_partition].append(sampleY)
                if current_partition == 0:
                    i_sample_within_batch += 1
            if i_sample_within_batch == batch_size:
                break
        if current_partition is None:
            partX = torch.cat(partX)
            partY = torch.cat(partY)
            return partX, partY
        for c_part in partX:
            partX[c_part] = torch.cat(partX[c_part])
            partY[c_part] = torch.cat(partY[c_part])
        return partX, partY
            
    
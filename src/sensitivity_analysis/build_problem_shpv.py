import numpy as np
import augmentations

from . import augmentation_setting
from . import utils

erase_params_dict = augmentation_setting._erase_params_dict
sharp_const_params_dict = augmentation_setting._sharp_const_params_dict
roll_params_dict = augmentation_setting._roll_params_dict
grayscale_params_dict = augmentation_setting._grayscale_params_dict
gaussian_blur_params_dict = augmentation_setting._gaussian_blur_params_dict

brightness_params_dict = augmentation_setting._brightness_params_dict
contrast_params_dict = augmentation_setting._contrast_params_dict
saturation_params_dict = augmentation_setting._saturation_params_dict
hue_params_dict = augmentation_setting._hue_params_dict
hflip_params_dict = augmentation_setting._hflip_params_dict
rotation_params_dict = augmentation_setting._rotation_params_dict
elb_params_dict = augmentation_setting._elb_params_dict

class_variable_name = utils._class_variable_name
permutation_variable_name = utils._permutation_variable_name
partition_variable_name = utils._partition_variable_name

def single_update_shpv_config(
    problem,
    Vtheta0_dict,
    transform_functions_list,
    transform_function,
    Vi_name,
    Vi_bounds=None,
    Vi_proba=0.5,
    Vi_dist=None,
    Vtheta_names=None,
    Vtheta_bounds=None,
    Vtheta_dists=None,
    Vtheta_0=None
):
    assert (Vi_bounds is not None) or ((Vtheta_names is not None) and (Vtheta_bounds is not None))
    num_variables_local = 0
    if Vi_bounds is not None:
        num_variables_local += 1
    if Vi_dist is None:
        Vi_dist = 'uniform'
    if Vtheta_names is not None:
        assert len(Vtheta_names) == len(Vtheta_bounds)
        num_variables_local += len(Vtheta_names)
        if Vtheta_dists is None:
            Vtheta_dists = ['uniform']*len(Vtheta_names)
        #for i, name in enumerate(Vtheta_names):
        #    Vtheta0_dict[name] = Vtheta_0[i]
        Vtheta0_dict[Vi_name] = Vtheta_0
            
        
    problem['num_vars'] = problem.get('num_vars', 0) + num_variables_local
    if Vi_bounds is not None:
        problem['names'] = problem.get('names', []) + [Vi_name]
        problem['bounds'] = problem.get('bounds', []) + [Vi_bounds]
        problem['groups'] = problem.get('groups', []) + [Vi_name]
    problem['dists'] = problem.get('dists', []) + [Vi_dist]
    if Vtheta_names is not None:
        problem['names'] = problem.get('names', []) + Vtheta_names
        problem['bounds'] = problem.get('bounds', []) + Vtheta_bounds
        problem['groups'] = problem.get('groups', []) + [Vi_name]*len(Vtheta_names)
        problem['dists'] = problem.get('dists', []) + Vtheta_dists
    
    theta_indices = problem.get('theta_indices', [0])
    theta_indices.append(theta_indices[-1]+num_variables_local)
    problem['theta_indices'] = theta_indices
    
    if transform_function is not None:
        transform_functions_list.append(transform_function)
    return problem, transform_functions_list, Vtheta0_dict

def update_with_augmentation_set_1(
    problem,
    transform_functions_list,
    image_shape,
    Vi_groups,
    Vtheta0_dict,
    Vi_proba=0.5
):
    
    channels, height, width = image_shape

    # 1) erase
    
    erase_h_offset = height // erase_params_dict['icoef_h_offset']
    erase_w_offset = width // erase_params_dict['icoef_w_offset']
    
    erase_center_y_bounds = (erase_h_offset, height-erase_h_offset)
    erase_center_x_bounds = (erase_w_offset, width-erase_w_offset)
    erase_scale_bound = erase_params_dict['scale_bounds']
    erase_log_ratio_bounds = erase_params_dict['log_ratio_bounds']
    
    Vi_group_name = erase_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = erase_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.erasing_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vi_dist='uniform', ### ????
        Vtheta_names=[
            'erase_center_y', 'erase_center_x', 'erase_scale', 'erase_log_ratio'
        ],
        Vtheta_bounds=[
            erase_center_y_bounds, erase_center_x_bounds, erase_scale_bound,
            erase_log_ratio_bounds
        ],
        Vtheta_dists=['integers', 'integers', 'uniform', 'uniform'],
        Vtheta_0=Vtheta_0
    )
    
    # 2) sharpness_const
    Vi_group_name = sharp_const_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = sharp_const_params_dict['Vtheta_0']
        
    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.sharpness_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vi_dist='uniform', ### ????
        Vtheta_names=[],
        Vtheta_bounds=[],
        Vtheta_dists=[],
        Vtheta_0=Vtheta_0
    )

    # 3) rolling
    roll_h_offset = height // roll_params_dict['icoef_h_offset']
    roll_w_offset = width // roll_params_dict['icoef_w_offset']
    
    roll_y_bounds = (-roll_h_offset, roll_h_offset)
    roll_x_bounds = (-roll_w_offset, roll_w_offset)
    
    Vi_group_name = roll_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = roll_params_dict['Vtheta_0']
    
    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.rolling_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vi_dist='uniform', ### ????
        Vtheta_names=['roll_y', 'roll_x'],
        Vtheta_bounds=[roll_y_bounds, roll_x_bounds],
        Vtheta_dists=['integers', 'integers'],
        Vtheta_0=Vtheta_0
    )
    
    # 4) grayscaling
    
    Vi_group_name = grayscale_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = grayscale_params_dict['Vtheta_0']
    
    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        lambda x: augmentations.grayscale_transform(x, num_output_channels=channels),
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vi_dist='uniform', ### ????
        Vtheta_names=[],
        Vtheta_bounds=[],
        Vtheta_dists=[],
        Vtheta_0=Vtheta_0
    )
    
    # 5) gaussian_blur
    
    Vi_group_name = gaussian_blur_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = gaussian_blur_params_dict['Vtheta_0']
    
    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.gaussian_blur_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vi_dist='uniform', ### ????
        Vtheta_names=['gblur_sigma'],
        Vtheta_bounds=[gaussian_blur_params_dict['sigma_bounds']],
        Vtheta_dists=['uniform'],
        Vtheta_0=Vtheta_0
    )

    return problem, transform_functions_list, Vi_groups, Vtheta0_dict


def update_with_augmentation_set_2(
    problem,
    transform_functions_list,
    image_shape,
    Vi_groups,
    Vtheta0_dict,
    Vi_proba=0.5
):

    # 1) brightness
    Vi_group_name = brightness_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = brightness_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.brightness_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vi_dist='uniform',
        Vtheta_names=['brightness_factor'],
        Vtheta_bounds=[brightness_params_dict['factor_bounds']],
        Vtheta_0=Vtheta_0
    )

    # 2) contrast

    Vi_group_name = contrast_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = contrast_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.contrast_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vtheta_names=['contrast_factor'],
        Vtheta_bounds=[contrast_params_dict['factor_bounds']],
        Vtheta_0=Vtheta_0
    )


    # 3) saturation

    Vi_group_name = saturation_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = saturation_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.saturation_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vtheta_names=['saturation_factor'],
        Vtheta_bounds=[saturation_params_dict['factor_bounds']],
        Vtheta_0=Vtheta_0
    )

    # 4) hue

    Vi_group_name = hue_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = hue_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.hue_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vtheta_names=['hue_factor'],
        Vtheta_bounds=[hue_params_dict['factor_bounds']],
        Vtheta_0=Vtheta_0
    )

    # 5) horizontal flip

    Vi_group_name = hflip_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = hflip_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.horizontal_flip_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vtheta_names=[],
        Vtheta_bounds=[],
        Vtheta_0=Vtheta_0
    )

    # 6) custom rotation

    Vi_group_name = rotation_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = rotation_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.custom_rotation_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vtheta_names=['crt_angle'],
        Vtheta_bounds=[rotation_params_dict['angle_bounds']],
        Vtheta_0=Vtheta_0
    )

    # 7) elliptic local blur

    elbt_a_bounds = elb_params_dict['a_bounds']
    elbt_b_bounds = elb_params_dict['b_bounds']
    elbt_shift_x_bounds = elb_params_dict['shift_x_bounds']
    elbt_shift_y_bounds = elb_params_dict['shift_y_bounds']
    elbt_angle = elb_params_dict['angle_bounds']

    Vi_group_name = elb_params_dict['Vi_group_name']
    Vi_groups[Vi_group_name] = Vi_proba
    Vtheta_0 = elb_params_dict['Vtheta_0']

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        augmentations.elliptic_local_blur_transform,
        Vi_name=Vi_group_name,
        Vi_bounds=(0, 1),
        Vi_proba=Vi_proba,
        Vtheta_names=[
            'elb_a', 'elb_b', 'elb_shift_x', 'elb_shift_y', 'elb_angle'
        ],
        Vtheta_bounds=[
            elbt_a_bounds, elbt_b_bounds, elbt_shift_x_bounds,
            elbt_shift_y_bounds, elbt_angle
        ],
        Vtheta_0=Vtheta_0
    )
    return problem, transform_functions_list, Vi_groups, Vtheta0_dict

def build_augmented_classification_problem(
    augmentation_set_number,
    image_shape,
    num_classes,
    p_aug_set1=0.5,
    p_aug_set2=0.5,
    use_permutation_variable=False,
    use_partition_variable=False
):
    assert augmentation_set_number in [1, 2, 3]
    problem = {}
    transform_functions_list = []
    Vi_proba_list = []
    Vi_groups = {}
    Vtheta0_dict = {}
    
    if augmentation_set_number in [1, 3]:
        problem, transform_functions_list, Vi_groups, Vtheta0_dict = update_with_augmentation_set_1(
            problem, transform_functions_list, image_shape, Vi_groups, Vtheta0_dict, p_aug_set1
        )
    if augmentation_set_number in [2, 3]:
        problem, transform_functions_list, Vi_groups, Vtheta0_dict = update_with_augmentation_set_2(
            problem, transform_functions_list, image_shape, Vi_groups, Vtheta0_dict, p_aug_set2
        )
        
    if use_permutation_variable:
        problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
            problem,
            Vtheta0_dict,
            transform_functions_list,
            transform_function=None,
            Vi_name='permutation',
            Vi_bounds=(0, 1),
            Vi_proba=None,
            Vi_dist='uniform'
            #Vtheta_names=['perm_var'],
            #Vtheta_bounds=[(0, 1)]
        )

    problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
        problem,
        Vtheta0_dict,
        transform_functions_list,
        transform_function=None,
        Vi_name='class',
        Vi_bounds=(0, num_classes),
        Vi_proba=None,
        Vi_dist='integers'
        #Vtheta_names=[],
        #Vtheta_bounds=[]
    )
    
    if use_partition_variable:
        problem, transform_functions_list, Vtheta0_dict = single_update_shpv_config(
            problem,
            Vtheta0_dict,
            transform_functions_list,
            transform_function=None,
            Vi_name='partition',
            Vi_bounds=(0, 1),
            Vi_proba=None,
            Vi_dist='uniform'
            #Vtheta_names=['perm_var'],
            #Vtheta_bounds=[(0, 1)]
        )
    problem['num_classes'] = num_classes
    
    return problem, transform_functions_list, Vi_groups, Vtheta0_dict
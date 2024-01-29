import augmentations
import numpy as np

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


def single_update_si_config(
    problem,
    transform_functions_list,
    threshold_list,
    transform_function,
    Vi_name,
    Vi_bounds=None,
    Vi_threshold=0.5,
    Vtheta_names=None,
    Vtheta_bounds=None
):
    assert (Vi_bounds is not None) or ((Vtheta_names is not None) and (Vtheta_bounds is not None))
    num_variables_local = 0
    if Vi_bounds is not None:
        num_variables_local += 1
    if Vtheta_names is not None:
        assert len(Vtheta_names) == len(Vtheta_bounds)
        num_variables_local += len(Vtheta_names)
    problem['num_vars'] = problem.get('num_vars', 0) + num_variables_local
    if Vi_bounds is not None:
        problem['names'] = problem.get('names', []) + [Vi_name]
        problem['bounds'] = problem.get('bounds', []) + [Vi_bounds]
        problem['groups'] = problem.get('groups', []) + [Vi_name]
        threshold_list.append(Vi_threshold)
    if Vtheta_names is not None:
        problem['names'] = problem.get('names', []) + Vtheta_names
        problem['bounds'] = problem.get('bounds', []) + Vtheta_bounds
        problem['groups'] = problem.get('groups', []) + [Vi_name]*len(Vtheta_names)
    
    theta_indices = problem.get('theta_indices', [0])
    theta_indices.append(theta_indices[-1]+num_variables_local)
    problem['theta_indices'] = theta_indices
    
    if transform_function is not None:
        transform_functions_list.append(transform_function)
    return problem, transform_functions_list, threshold_list

def update_with_augmentation_set_1(
    problem,
    transform_functions_list,
    threshold_list,
    image_shape,
    p_threshold=0.5,
    use_switch_variables=False
):
    channels, height, width = image_shape
    
    ### 1. erasing
    
    erase_h_offset = height // erase_params_dict['icoef_h_offset']
    erase_w_offset = width // erase_params_dict['icoef_w_offset']
    
    erase_center_y_bounds = (erase_h_offset, height-erase_h_offset)
    erase_center_x_bounds = (erase_w_offset, width-erase_w_offset)
    erase_scale_bound = erase_params_dict['scale_bounds']
    erase_log_ratio_bounds = erase_params_dict['log_ratio_bounds']
    
    Vi_name = erase_params_dict['Vi_group_name']
    
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.erasing_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=[
            'erase_center_y', 'erase_center_x', 'erase_scale', 'erase_log_ratio'
        ],
        Vtheta_bounds=[
            erase_center_y_bounds, erase_center_x_bounds, erase_scale_bound,
            erase_log_ratio_bounds
        ]
    )
    
    # 2) sharpness_const
    Vi_name = sharp_const_params_dict['Vi_group_name']
    
    if use_switch_variables:
        local_sharpness_transform = augmentations.sharpness_transform
    else:
        local_sharpness_transform = lambda img, p: (
            img if p > p_threshold else augmentations.sharpness_transform(img)
        )
        
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        local_sharpness_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1),# if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=[],
        Vtheta_bounds=[]
    )
    
    # 3) rolling
    roll_h_offset = height // roll_params_dict['icoef_h_offset']
    roll_w_offset = width // roll_params_dict['icoef_w_offset']
    
    roll_y_bounds = (-roll_h_offset, roll_h_offset)
    roll_x_bounds = (-roll_w_offset, roll_w_offset)
    
    Vi_name = roll_params_dict['Vi_group_name']
    
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.rolling_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=['roll_y', 'roll_x'],
        Vtheta_bounds=[roll_y_bounds, roll_x_bounds]
    )
    
    # 4) grayscaling
    
    Vi_name = grayscale_params_dict['Vi_group_name']
    
    if use_switch_variables:
        local_grayscaling_transform = lambda img: augmentations.grayscale_transform(
            img, num_output_channels=channels
        )
    else:
        local_grayscaling_transform = lambda img, p: (
            img if p > p_threshold else augmentations.grayscale_transform(
                img, num_output_channels=channels
            )
        )
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        local_grayscaling_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1),# if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=[],
        Vtheta_bounds=[]
    )
    
    # 5) gaussian_blur
    Vi_name = gaussian_blur_params_dict['Vi_group_name']
    
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.gaussian_blur_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=['gblur_sigma'],
        Vtheta_bounds=[gaussian_blur_params_dict['sigma_bounds']]
    )

    return problem, transform_functions_list, threshold_list


def update_with_augmentation_set_2(
    problem,
    transform_functions_list,
    threshold_list,
    image_shape,
    p_threshold=None,
    use_switch_variables=False
):
    # 1) brightness
    Vi_name = brightness_params_dict['Vi_group_name']
    
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.brightness_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=['brightness_factor'],
        Vtheta_bounds=[brightness_params_dict['factor_bounds']]
    )
    
    # 2) contrast

    Vi_name = contrast_params_dict['Vi_group_name']

    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.contrast_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=['contrast_factor'],
        Vtheta_bounds=[contrast_params_dict['factor_bounds']]
    )
    
    # 3) saturation

    Vi_name = saturation_params_dict['Vi_group_name']

    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.saturation_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=['saturation_factor'],
        Vtheta_bounds=[saturation_params_dict['factor_bounds']]
    )
    
    # 4) hue

    Vi_name = hue_params_dict['Vi_group_name']

    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.hue_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=['hue_factor'],
        Vtheta_bounds=[hue_params_dict['factor_bounds']]
    )
    
    # 5) horizontal flip

    Vi_name = hflip_params_dict['Vi_group_name']

    if use_switch_variables:
        local_hflip_transform = lambda img: augmentations.horizontal_flip_transform(img)
    else:
        local_hflip_transform = lambda img, p: (
            img if p > p_threshold else augmentations.horizontal_flip_transform(img)
        )
        
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        local_hflip_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1),# if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=[], # if use_switch_variables else ['hflip_p'],
        Vtheta_bounds=[] # if use_switch_variables else [(0, 1)]
    )
    
    # 6) custom rotation

    Vi_name = rotation_params_dict['Vi_group_name']

    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.custom_rotation_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=['crt_angle'],
        Vtheta_bounds=[rotation_params_dict['angle_bounds']]
    )
    
    # 7) elliptic local blur

    elbt_a_bounds = elb_params_dict['a_bounds']
    elbt_b_bounds = elb_params_dict['b_bounds']
    elbt_shift_x_bounds = elb_params_dict['shift_x_bounds']
    elbt_shift_y_bounds = elb_params_dict['shift_y_bounds']
    elbt_angle = elb_params_dict['angle_bounds']

    Vi_name = elb_params_dict['Vi_group_name']
    
    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        augmentations.elliptic_local_blur_transform,
        Vi_name=Vi_name,
        Vi_bounds=(0, 1) if use_switch_variables else None,
        Vi_threshold=p_threshold,
        Vtheta_names=[
            'elb_a', 'elb_b', 'elb_shift_x', 'elb_shift_y', 'elb_angle'
        ],
        Vtheta_bounds=[
            elbt_a_bounds, elbt_b_bounds, elbt_shift_x_bounds,
            elbt_shift_y_bounds, elbt_angle
        ]
    )
    return problem, transform_functions_list, threshold_list

def build_augmented_classification_problem(
    augmentation_set_number,
    image_shape,
    p_aug_set1=0.5,
    p_aug_set2=0.5,
    use_permutation_variable=False,
    use_switch_variables=False,
    use_partition_variable=False # train/test selector
):
    assert augmentation_set_number in [1, 2, 3]
    problem = {}
    transform_functions_list = []
    threshold_list = []
    
    if augmentation_set_number in [1, 3]:
        problem, transform_functions_list, threshold_list = update_with_augmentation_set_1(
            problem, transform_functions_list, threshold_list, image_shape, p_aug_set1, use_switch_variables
        )
    if augmentation_set_number in [2, 3]:
        problem, transform_functions_list, threshold_list = update_with_augmentation_set_2(
            problem, transform_functions_list, threshold_list, image_shape, p_aug_set2, use_switch_variables
        )

    if use_permutation_variable:
        problem, transform_functions_list, threshold_list = single_update_si_config(
            problem,
            transform_functions_list,
            threshold_list,
            transform_function=None,
            Vi_name=permutation_variable_name,
            Vi_bounds=(0, 1),
            Vi_threshold=None,
            #Vtheta_names=['perm_var'],
            #Vtheta_bounds=[(0, 1)]
        )

    problem, transform_functions_list, threshold_list = single_update_si_config(
        problem,
        transform_functions_list,
        threshold_list,
        transform_function=None,
        Vi_name=class_variable_name,
        Vi_bounds=(0, 1),
        Vi_threshold=None,
        #Vtheta_names=[],
        #Vtheta_bounds=[]
    )
    if use_partition_variable:
        problem, transform_functions_list, threshold_list = single_update_si_config(
            problem,
            transform_functions_list,
            threshold_list,
            transform_function=None,
            Vi_name=partition_variable_name,
            Vi_bounds=(0, 1),
            Vi_threshold=None
        )
    
    return problem, transform_functions_list, threshold_list
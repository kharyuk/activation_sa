import numpy as np
import augmentations

from . import augmentation_setting

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



def single_update_cont_config(
    problem,
    transform_functions_list,
    transform_function,
    Vi_name,
    Vtheta_names=None,
    Vtheta_bounds=None,
    Vtheta_dists=None
):
    assert (Vtheta_names is None) or ((Vtheta_names is not None) and (Vtheta_bounds is not None))
    num_variables_local = 0
    if Vtheta_names is not None:
        assert len(Vtheta_names) == len(Vtheta_bounds)
        num_variables_local += len(Vtheta_names)
        if (Vtheta_dists is None) or (len(Vtheta_dists) == 0):
            Vtheta_dists = ['uniform']*len(Vtheta_names)
        
    problem['num_augs'] = problem.get('num_augs', 0) + 1
    problem['aug_names'] = problem.get('aug_names', []) + [Vi_name]
    
    if (Vtheta_names is not None) and len(Vtheta_names) > 0:
        problem['aug_types'] = problem.get('aug_types', []) + ['param']
        problem['names'] = problem.get('names', []) + Vtheta_names
        problem['bounds'] = problem.get('bounds', []) + Vtheta_bounds
        problem['groups'] = problem.get('groups', []) + [Vi_name]*len(Vtheta_names)
        problem['dists'] = problem.get('dists', []) + Vtheta_dists
        
        theta_indices = problem.get('theta_indices', [0])
        theta_indices.append(theta_indices[-1]+num_variables_local)
        problem['theta_indices'] = theta_indices
    
    else:
        problem['aug_types'] = problem.get('aug_types', []) + ['nonpar']
        
    transform_functions_list.append(transform_function)
    return problem, transform_functions_list

def update_with_augmentation_set_1(
    problem, transform_functions_list, image_shape
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

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.erasing_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=[
            'erase_center_y', 'erase_center_x', 'erase_scale', 'erase_log_ratio'
        ],
        Vtheta_bounds=[
            erase_center_y_bounds, erase_center_x_bounds, erase_scale_bound,
            erase_log_ratio_bounds
        ],
        Vtheta_dists=['integers', 'integers', 'uniform', 'uniform'],
    )
    
    # 2) sharpness_const
    Vi_group_name = sharp_const_params_dict['Vi_group_name']
        
    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.sharpness_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=[],
        Vtheta_bounds=[],
        Vtheta_dists=[]
    )

    # 3) rolling
    
    roll_h_offset = height // roll_params_dict['icoef_h_offset']
    roll_w_offset = width // roll_params_dict['icoef_w_offset']
    
    roll_y_bounds = (-roll_h_offset, roll_h_offset)
    roll_x_bounds = (-roll_w_offset, roll_w_offset)
    
    Vi_group_name = roll_params_dict['Vi_group_name']
    
    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.rolling_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=['roll_y', 'roll_x'],
        Vtheta_bounds=[roll_y_bounds, roll_x_bounds],
        Vtheta_dists=['integers', 'integers'],
    )
    
    # 4) grayscaling
    
    Vi_group_name = grayscale_params_dict['Vi_group_name']
    
    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        lambda x: augmentations.grayscale_transform(x, num_output_channels=channels),
        Vi_name=Vi_group_name,
        Vtheta_names=[],
        Vtheta_bounds=[],
        Vtheta_dists=[]
    )
    
    # 5) gaussian_blur
    
    Vi_group_name = gaussian_blur_params_dict['Vi_group_name']    
    
    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.gaussian_blur_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=['gblur_sigma'],
        Vtheta_bounds=[gaussian_blur_params_dict['sigma_bounds']],
        Vtheta_dists=['uniform']
    )

    return problem, transform_functions_list

def update_with_augmentation_set_2(
    problem, transform_functions_list, image_shape
):

    # 1) brightness
    Vi_group_name = brightness_params_dict['Vi_group_name']

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.brightness_transform,
        Vi_name=Vi_group_name,        
        Vtheta_names=['brightness_factor'],
        Vtheta_bounds=[brightness_params_dict['factor_bounds']],
        Vtheta_dists=['uniform'],
    )

    # 2) contrast

    Vi_group_name = contrast_params_dict['Vi_group_name']

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.contrast_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=['contrast_factor'],
        Vtheta_bounds=[contrast_params_dict['factor_bounds']],
        Vtheta_dists=['uniform'],
    )


    # 3) saturation
    
    Vi_group_name = saturation_params_dict['Vi_group_name']

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.saturation_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=['saturation_factor'],
        Vtheta_bounds=[saturation_params_dict['factor_bounds']],
        Vtheta_dists=['uniform'],
    )

    # 4) hue

    Vi_group_name = hue_params_dict['Vi_group_name']

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.hue_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=['hue_factor'],
        Vtheta_bounds=[hue_params_dict['factor_bounds']],
        Vtheta_dists=['uniform'],
    )

    # 5) horizontal flip

    Vi_group_name = hflip_params_dict['Vi_group_name']

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.horizontal_flip_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=[],
        Vtheta_bounds=[]
    )

    # 6) custom rotation

    Vi_group_name = rotation_params_dict['Vi_group_name']

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.custom_rotation_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=['crt_angle'],
        Vtheta_bounds=[rotation_params_dict['angle_bounds']],
        Vtheta_dists=['uniform'],
    )

    # 7) elliptic local blur

    elbt_a_bounds = elb_params_dict['a_bounds']
    elbt_b_bounds = elb_params_dict['b_bounds']
    elbt_shift_x_bounds = elb_params_dict['shift_x_bounds']
    elbt_shift_y_bounds = elb_params_dict['shift_y_bounds']
    elbt_angle = elb_params_dict['angle_bounds']

    Vi_group_name = elb_params_dict['Vi_group_name']

    problem, transform_functions_list = single_update_cont_config(
        problem,
        transform_functions_list,
        augmentations.elliptic_local_blur_transform,
        Vi_name=Vi_group_name,
        Vtheta_names=[
            'elb_a', 'elb_b', 'elb_shift_x', 'elb_shift_y', 'elb_angle'
        ],
        Vtheta_bounds=[
            elbt_a_bounds, elbt_b_bounds, elbt_shift_x_bounds,
            elbt_shift_y_bounds, elbt_angle
        ],
        Vtheta_dists=['integers', 'integers', 'integers', 'integers', 'uniform'],
    )
    return problem, transform_functions_list

def build_augmented_classification_problem(
    augmentation_set_number,
    image_shape,
    num_classes
):
    assert augmentation_set_number in [1, 2, 3]
    problem = {}
    transform_functions_list = []
    
    if augmentation_set_number in [1, 3]:
        problem, transform_functions_list = update_with_augmentation_set_1(
            problem, transform_functions_list, image_shape
        )
    if augmentation_set_number in [2, 3]:
        problem, transform_functions_list = update_with_augmentation_set_2(
            problem, transform_functions_list, image_shape
        )
        
    problem['num_classes'] = num_classes
    return problem, transform_functions_list
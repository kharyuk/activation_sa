import numpy as np

_augmentation_set1_variable_names = [
    'erasing_transform',
    'sharpness_const',
    'rolling_transform',
    'grayscaling_transform',
    'gaussian_blur'
]
_augmentation_set2_variable_names = [
    'brightness_transform',
    'contrast_transform',
    'saturation_transform',
    'hue_transform',
    'hflip',
    'rotation',
    'elliptic_local_blur',
]

_erase_params_dict = {
    'Vi_group_name': 'erasing',
    'Vtheta_0': np.array([1, 1, 0., 0.]),
    'icoef_h_offset': 5,
    'icoef_w_offset': 5,
    'scale_bounds': (0.02, 0.33),
    'log_ratio_bounds': (np.log(0.3), np.log(3.3)),
}

_sharp_const_params_dict = {
    'Vi_group_name': 'sharpness_const',
    'Vtheta_0': np.array([])
}

_roll_params_dict = {
    'Vi_group_name': 'rolling',
    'Vtheta_0': np.array([0., 0.]),
    'icoef_h_offset': 5,
    'icoef_w_offset': 5,
}

_grayscale_params_dict = {
    'Vi_group_name': 'grayscaling',
    'Vtheta_0': np.array([])
}

_gaussian_blur_params_dict = {
    'Vi_group_name': 'gaussian_blur',
    'Vtheta_0': np.array([0.]),
    'sigma_bounds': (0.1, 2.)
}

_brightness_params_dict = {
    'Vi_group_name': 'brightness',
    'Vtheta_0': np.array([1.]),
    'factor_bounds': (0.1, 2.5)
}

_contrast_params_dict = {
    'Vi_group_name': 'contrast',
    'Vtheta_0': np.array([1.]),
    'factor_bounds': (0.01, 3.5)
}

_saturation_params_dict = {
    'Vi_group_name': 'saturation',
    'Vtheta_0': np.array([1.]),
    'factor_bounds': (0.01, 3.5)
}

_hue_params_dict = {
    'Vi_group_name': 'hue',
    'Vtheta_0': np.array([1.]),
    'factor_bounds': (-0.5, 0.5)
}

_hflip_params_dict = {
    'Vi_group_name': 'hflip',
    'Vtheta_0': np.array([])
}

_rotation_params_dict = {
    'Vi_group_name': 'rotation',
    'Vtheta_0': np.array([0.]),
    'angle_bounds': (-10, 10)
}

_elb_params_dict = {
    'Vi_group_name': 'elliptic_local_blur',
    'Vtheta_0': np.array([0, 0, 0, 0, 0]),
    'a_bounds': (10, 32),
    'b_bounds': (10, 32),
    'shift_x_bounds': (-32, 32),
    'shift_y_bounds': (-32, 32),
    'angle_bounds': (-15, 15)
}

_listed_params_dicts = [
    _erase_params_dict,
    _sharp_const_params_dict,
    _roll_params_dict,
    _grayscale_params_dict,
    _gaussian_blur_params_dict,
    _brightness_params_dict,
    _contrast_params_dict,
    _saturation_params_dict,
    _hue_params_dict,
    _hflip_params_dict,
    _rotation_params_dict,
    _elb_params_dict
]

def get_dict(dict_list, key, value):
    for d in dict_list:
        if (d[key].startswith(value)) or (value.startswith(d[key])):
            return d
    raise ValueError
    

def get_group_variables_indices(
    augmentation_set_number,
    use_permutation_variable,
    use_class_variable,
    use_partition_variable
):
    variable_names = []
    if augmentation_set_number in [1, 3]:
        variable_names += _augmentation_set1_variable_names
    if augmentation_set_number in [2, 3]:
        variable_names += _augmentation_set2_variable_names
    ind = 0
    group_indices = [ind]
    for vname in variable_names:
        d = get_dict(_listed_params_dicts, 'Vi_group_name', vname)
        offset = (len(d)-2) + 1
        ind += offset
        group_indices.append(ind)
    for bval in [
        use_permutation_variable,
        use_class_variable,
        use_partition_variable
    ]:
        ind += 1
        group_indices.append(ind)
    return group_indices
        

def get_augmentation_transforms_names(
    augmentation_set_number,
    use_permutation_variable,
    use_class_variable,
    use_partition_variable
):
    rv = []
    if augmentation_set_number in [1, 3]:
        rv += _augmentation_set1_variable_names
    if augmentation_set_number in [2, 3]:
        rv += _augmentation_set2_variable_names
    if use_permutation_variable:
        rv.append('permutation')
    if use_class_variable:
        rv.append('class')
    if use_partition_variable:
        rv.append('partition')
    return rv
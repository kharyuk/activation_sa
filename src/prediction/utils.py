
import preparation.single_unit
import preparation.visualize

_shorten_aug_aux_names_dict = dict(
    **preparation.visualize._shorten_augnames_dict,
    **preparation.visualize._shorten_variables_dict,
)


def get_shortened_variable_names_dict(
    augmentation_set_numbers_list,
    extract_auxilliary_names=True,
):
    augmentation_names_dict = preparation.single_unit.extract_augmentation_names_dict(
        augmentation_set_numbers_list,
        extract_auxilliary_names=extract_auxilliary_names,
    )
    #augmentation_names = functools.reduce(
    #    lambda x, y: x+y, augmentation_names_dict.values()
    #)
    for augmentation_set_number in augmentation_set_numbers_list:
        augmentation_names_dict[augmentation_set_number] = list(
            map(
                lambda x: _shorten_aug_aux_names_dict[x],
                augmentation_names_dict[augmentation_set_number]
            )
        )
    return augmentation_names_dict

def get_shortened_variable_names_single_augset(
    augmentation_set_number,
    extract_auxilliary_names=True,
):
    augmentation_names = get_shortened_variable_names_dict(
        [augmentation_set_number],
        extract_auxilliary_names=False,
    )
    augmentation_names = augmentation_names[augmentation_set_number]
    if extract_auxilliary_names:
        augmentation_and_auxilliary_names = get_shortened_variable_names_dict(
            [augmentation_set_number],
            extract_auxilliary_names=True,
        )
        augmentation_and_auxilliary_names = augmentation_and_auxilliary_names[augmentation_set_number]
    else:
        augmentation_and_auxilliary_names = augmentation_names
    return augmentation_names, augmentation_and_auxilliary_names

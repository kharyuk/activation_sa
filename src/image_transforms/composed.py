import math
import copy

# duplicate ? -> sensitivity_analysis/{utils; sobol}

def perm_recursion(ind, sequence, result):
    N = len(sequence)
    if N == 0:
        return []
    #sequence.sort()
    cur_fact = math.factorial(N-1)
    new_ind, rem = ind // cur_fact, ind % cur_fact
    result.append(sequence[new_ind])
    del sequence[new_ind];
    if rem == 0:
        return result+sequence
    return perm_recursion(rem, sequence, result)

def permute_by_index(ind, sequence):
    N = len(sequence)
    assert 0 <= ind < math.factorial(N)
    return perm_recursion(ind, copy.copy(sequence), [])

def composed_transform(input_batch, transform_functions_list, transform_parameters):
    N_transforms = len(transform_functions_list)
    permutation_ind = None
    if N_transforms < len(transform_parameters):
        permutation_ind = int(round(N_transforms*transform_parameters[-1]))
        transform_parameters = transform_parameters[:-1]
    paired_tfun_p = list(zip(transform_functions_list, transform_parameters))
    if permutation_ind is not None:
        paired_tfun_p = permute_by_index(
            permutation_ind, paired_tfun_p
        )
    output_batch = input_batch
    for (transform, p) in paired_tfun_p:
        if p >= 0.5:
            output_batch = transform(output_batch)
    return output_batch
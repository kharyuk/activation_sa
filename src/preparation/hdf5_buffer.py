import h5py
import os
import numpy as np

import hdf5_utils


def extract_cyclic_ind(I, L, A):
    '''
    input:
        I = index (pointer)
        L = limit size,
        A = length of data to be written
    output:
        rv = list of chronological index pairs
        P = number of fullfills
    '''
    rv = []
    M, N = A // L, A % L
    if N > 0:
        if I+N <= L:
            rv += [(I, I+N)]
        else:
            rv += [(I, L), (0, N+I-L)]
    if (N == 0) and (M > 0):
        if I == 0:
            rv += [(0, L)]
        else:
            rv += [(I, L), (0, I)]
    elif M > 0: # N > 0
        Q = rv[-1][1]
        D = L-N
        if Q+D <= L:
            rv = [(Q, Q+D)] + rv
        else:
            rv = [(Q, L), (0, Q+D-L)] + rv
    rv = list(filter(lambda x: x[0] < x[1], rv))
    for i in range(len(rv)-1, 0, -1):
        if rv[i-1][1] == rv[i][0]:
            rv[i-1] = (rv[i-1][0], rv[i][1])
            del rv[i]
    P = M + (I+N)//L
    return rv, P

def init_hdf5_buffer(
    hdf5_file_path,
    Nvariables,
    Nouter_samples,
    Ninner_samples,
    Npermutations,
    reinit_existing=True
):
    block_size = Nvariables*Nouter_samples*Ninner_samples
    #buffer_capacity = block_size*Npermutations
    dirname, _ = os.path.split(hdf5_file_path)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    if os.path.isfile(hdf5_file_path):
        if reinit_existing:
            with h5py.File(hdf5_file_path, 'r') as activ:
                assert Npermutations == activ.attrs['num_blocks']
                assert block_size == activ.attrs['block_size']
            with h5py.File(hdf5_file_path, 'a') as activ:
                activ.attrs['block_pointer'] = 0
                activ.attrs['position_pointer'] = 0
                activ.attrs['fullfill_counter'] = 0
                activ.attrs['sigmas'] = np.ones((Npermutations, Nvariables), dtype='i')
                activ.attrs['sigmas_pointer'] = 0
            return hdf5_file_path
            
    with h5py.File(hdf5_file_path, 'w') as activ:
        activ.attrs.create('block_pointer', 0)
        activ.attrs.create('position_pointer', 0)
        activ.attrs.create('num_blocks', Npermutations)
        activ.attrs.create('block_size', block_size)
        activ.attrs.create('fullfill_counter', 0)
        #vp_group = activ.create_group('variables_permutation')
        #vp_group.create_dataset(
        #    name='sigmas',
        #    shape=(buffer_capacity, Nvariables)
        #)
        #shape = (buffer_capacity, Nvariables)
        shape = (Npermutations, Nvariables)
        activ.attrs.create('sigmas', data=np.ones(shape), shape=shape, dtype='i')
        activ.attrs.create('sigmas_pointer', 0)
    return hdf5_file_path

def fill_buffer(hdf5_dataset, data, ind):
    data_size = len(data)
    ll = [x[1]-x[0] for x in ind]
    L = sum(ll)
    if len(ind) == 1:
        hdf5_dataset[ind[0][0]:ind[0][1]] = data[-L:]
    elif len(ind) == 2:
        hdf5_dataset[ind[0][0]:ind[0][1]] = data[-L:-ll[1]]
        hdf5_dataset[ind[1][0]:ind[1][1]] = data[-ll[1]:]
    else:
        raise ValueError
    return ind[-1][1]

def fill_sigmas(hdf5_dataset, sigma):
    num_blocks = hdf5_dataset.attrs['num_blocks']
    #vp_group = activ['variables_permutation']
    #pointer = vp_group['pointer']
    #vp_group['sigmas'][pointer, :] = sigma
    #pointer += 1
    #vp_group['pointer'] = pointer % num_blocks
    sigmas_pointer = hdf5_dataset.attrs['sigmas_pointer']
    sigmas = hdf5_dataset.attrs['sigmas']
    sigmas[sigmas_pointer] = np.array(sigma, dtype='i')
    hdf5_dataset.attrs['sigmas'] = sigmas
    sigmas_pointer += 1
    hdf5_dataset.attrs['sigmas_pointer'] = sigmas_pointer % num_blocks
    return

def resave_buffer(
    hdf5_file_path,
    new_hdf5_file_path
):
    #buffer_capacity = block_size*Npermutations
    #dirname, _ = os.path.split(hdf5_file_path)
    with h5py.File(hdf5_file_path, 'r') as activ:
        num_blocks = activ.attrs['num_blocks']
        block_size = activ.attrs['block_size']
        block_pointer = activ.attrs['block_pointer']
        position_pointer = activ.attrs['position_pointer']
        fullfill_counter = activ.attrs['fullfill_counter']
        sigmas = activ.attrs['sigmas']# = np.ones((Npermutations, Nvariables), dtype='i')
        sigmas_pointer = activ.attrs['sigmas_pointer']
        sigmas_act = sigmas[:0]
        if block_pointer > 0:
            ind = 0
            offset = block_pointer*block_size
            sigmas_act = np.append(sigmas_act, sigmas[:sigmas_pointer], axis=0)
            for name in activ:
                #print(activ[name]['activations'][ind:ind+offset].shape)
                hdf5_utils.update_data_only_hdf5(
                    {name: activ[name]['activations'][ind:ind+offset]}, new_hdf5_file_path
                )
        #if fullfill_counter == 0:
        #    return new_hdf5_file_path
        ind = block_pointer*block_size
        i_s = sigmas_pointer
        if position_pointer > 0:
            ind += block_size
            i_s += 1
        if fullfill_counter == 0:
            offset = (block_pointer+1)*block_size
        else:
            offset = num_blocks*block_size
        if (ind < offset) and (ind < num_blocks*block_size):
            sigmas_act = np.append(sigmas_act, sigmas[i_s:], axis=0)
            for name in activ:
                hdf5_utils.update_data_only_hdf5(
                    {name: activ[name]['activations'][ind:ind+offset]}, new_hdf5_file_path
                )
    hdf5_utils.copy_attrs_hdf5(hdf5_file_path, new_hdf5_file_path)
    with h5py.File(new_hdf5_file_path, 'a') as activ:
        activ.attrs['sigmas'] = sigmas_act
    return new_hdf5_file_path

def iterate_attrs(activations_fnm, batch_size):
    with h5py.File(activations_fnm, 'a') as activ:
        block_pointer = activ.attrs['block_pointer']
        position_pointer = activ.attrs['position_pointer']
        num_blocks = activ.attrs['num_blocks']
        block_size = activ.attrs['block_size']
        
        buffer_capacity = block_size*num_blocks
        current_ind = block_pointer*block_size+position_pointer
        
        ind, P = extract_cyclic_ind(current_ind, buffer_capacity, batch_size)
        activ.attrs['fullfill_counter'] += P
        
        #current_ind = fill_buffer(current_group['activations'], outputs, ind)
        current_ind = ind[-1][1]
        block_pointer, position_pointer = current_ind // block_size, current_ind % block_size
        activ.attrs['block_pointer'] = block_pointer
        activ.attrs['position_pointer'] = position_pointer


def save_activations(activations_fnm, name, mod, inputs, outputs):
    # activations_fnm
    outputs = outputs.detach().cpu().numpy().astype('float32')
    batch_size = len(outputs)
    shape = outputs.shape[1:]
    outputs = np.reshape(outputs, (batch_size, -1))
    
    with h5py.File(activations_fnm, 'a') as activ:
        block_pointer = activ.attrs['block_pointer']
        position_pointer = activ.attrs['position_pointer']
        num_blocks = activ.attrs['num_blocks']
        block_size = activ.attrs['block_size']
        
        buffer_capacity = block_size*num_blocks
        current_ind = block_pointer*block_size+position_pointer
        
        if name in activ:
            current_group = activ[name]
        else:
            current_group = activ.create_group(name)
            current_group.create_dataset(
                name='activations',
                dtype=outputs.dtype,
                compression="gzip",
                chunks=True,
                shape=(buffer_capacity, ) + outputs.shape[1:]
            )
            current_group.attrs.create('shape', data=shape)
        
        ind, P = extract_cyclic_ind(current_ind, buffer_capacity, batch_size)
        activ.attrs['fullfill_counter'] += P
        
        current_ind = fill_buffer(current_group['activations'], outputs, ind)
        #block_pointer, position_pointer = current_ind // block_size, current_ind % block_size
        #activ.attrs['block_pointer'] = block_pointer
        #activ.attrs['position_pointer'] = position_pointer
 





import functools
import os
import gc
import h5py
import numpy as np


def copy_attrs_hdf5(path1, path2):
    with h5py.File(path2, 'a') as save_fd:
        with h5py.File(path1, 'r') as copy_fd:
            for attr_name in copy_fd.attrs:
                save_fd.attrs.create(
                    attr_name, data=copy_fd.attrs[attr_name]
                )
            for group_name in copy_fd:
                for attr_name in copy_fd[group_name].attrs:
                    save_fd[group_name].attrs.create(
                        attr_name, data=copy_fd[group_name].attrs[attr_name]
                    )
    return

def update_data_only_hdf5(data_dict, hdf5_file_path):
    '''
    no groups!
    '''
    with h5py.File(hdf5_file_path, 'a') as hdf5_file:
        for name, X in data_dict.items():
            if name in hdf5_file:
                current_group = hdf5_file[name]
                len_samples = current_group['activations'].shape[0]
                len_new_samples = len(X)
                current_group['activations'].resize(len_samples+len_new_samples, axis=0)
                current_group['activations'][-len_new_samples:] = X
            else:
                current_group = hdf5_file.create_group(name)
                current_group.create_dataset(
                    name='activations',
                    data=X,
                    compression="gzip",
                    chunks=True,
                    maxshape=(None, ) + X.shape[1:]
                )     
                
def cut_activations(load_path, save_path, N, remove_original=False, buffer_size=10000):
    with h5py.File(save_path, 'a') as activ2:
        with h5py.File(load_path, 'r') as activ1:
            for name in activ1:
                current_group1 = activ1[name]
                activations_shape = current_group1['activations'].shape
                len_a, activations_shape = activations_shape[0], activations_shape[1:]
                assert N < len_a

                current_group2 = activ2.create_group(name)
                current_group2.create_dataset(
                    name='activations',
                    data=None,
                    #dtype=outputs.dtype,
                    compression="gzip",
                    chunks=True,
                    maxshape=(N, ) + outputs.shape[1:]
                )
                current_group2.attrs.create('shape', data=current_group1.attrs['shape'])
                i = 0
                while i < N:
                    current_group2[i:i+buffer_size] = current_group1[i:i+buffer_size]
                    i += buffer_size
    if remove_original:
        os.remove(load_path)
    return save_path

def cut_activations2(load_path, save_path, N, remove_original=False, buffer_size=10000):
    if os.path.isfile(save_path):
        os.remove(save_path)
    with h5py.File(save_path, 'a') as activ2:
        with h5py.File(load_path, 'r') as activ1:
            for name in activ1:
                current_group1 = activ1[name]
                activations_shape = current_group1['activations'].shape
                len_a, activations_shape = activations_shape[0], activations_shape[1:]
                assert N < len_a
                
                i = 0
                offset = min(N, buffer_size)
                
                current_group2 = activ2.create_group(name)
                current_group2.create_dataset(
                    name='activations',
                    data=current_group1['activations'][i:i+offset],
                    #dtype=current_group1['activations'].dtype,
                    compression="gzip",
                    chunks=True,
                    maxshape=(N, ) + activations_shape
                )
                current_group2.attrs.create('shape', data=current_group1.attrs['shape'])
                i += offset
                
                while i < N:
                    len_samples = i
                    limit = min(N, i+buffer_size)
                    X = current_group1['activations'][i:limit]
                    len_new_samples = len(X)
                    current_group2['activations'].resize(
                        len_samples+len_new_samples, axis=0
                    )
                    current_group2['activations'][-len_new_samples:] = X
                    i += buffer_size
                    
    if remove_original:
        os.remove(load_path)
    return save_path
                
def save_activations(activations_fnm, name, mod, inputs, outputs):
    # activations_fnm
    outputs = outputs.detach().cpu().numpy().astype('float32')
    shape = outputs.shape[1:]
    len_b = len(outputs)
    outputs = np.reshape(outputs, (len_b, -1))
    with h5py.File(activations_fnm, 'a') as activ:
        if name in activ:
            current_group = activ[name]
            len_a = current_group['activations'].shape[0]
            current_group['activations'].resize(len_a+len_b, axis=0)
            current_group['activations'][-len_b:] = outputs
        else:
            current_group = activ.create_group(name)
            current_group.create_dataset(
                name='activations',
                data=outputs,
                #dtype=outputs.dtype,
                compression="gzip",
                chunks=True,
                maxshape=(None, ) + outputs.shape[1:]
            )
            current_group.attrs.create('shape', data=shape)


def get_values(
    values_path,
    module_names,
    values_name,
    preprocess_values_func=None,
):
    values = []
    with h5py.File(values_path, 'r') as vals:
        for i_mn, module_name in enumerate(module_names):
            current_group = vals[module_name]
            if isinstance(values_name, str):
                current_values = current_group[values_name][:]
            else:
                current_values = []
                for vname in values_name:
                    current_values.append(current_group[vname][:])
            act_shape = tuple(current_group.attrs['shape'])
            #act_shape = tuple(vals.attrs[module_name])
            if isinstance(values_name, str):
                current_values = current_values.reshape((-1, ) + act_shape)
            else:
                len_v = len(values_name)
                current_values = [x.reshape((len_v, -1) + act_shape) for x in current_values]
            values.append(current_values)
    if preprocess_values_func is not None:
        values = values_func(values)
    return values



def get_remove_dataset_parts(buf, removing_dataset_name, name, node):
    if isinstance(node, h5py.Dataset):
        dataset_name, group_name = name[::-1].split('/', 1)
        group_name, dataset_name = group_name[::-1], dataset_name[::-1]
        if dataset_name != removing_dataset_name:
            return
        buf.append(name)

def clean_dataset_part(path, removing_dataset_name):
    buf = []
    current_func = functools.partial(get_remove_dataset_parts, buf, removing_dataset_name)
    with h5py.File(path, 'r') as fd:
        fd.visititems(current_func)
        for p in buf:
            del fd[p]
            
def resav_dir(dirpath1, dirpath2):
    files = os.listdir(dirpath1)
    #for p in files:
    while len(files) > 0:
        p = files[0]
        sub_p = os.path.join(dirpath1, p)
        dest = os.path.join(dirpath2, p)
        if os.path.isdir(sub_p):
            sub_p_list = os.listdir(sub_p)
            os.makedirs(dest, exist_ok=True)
            files += [os.path.join(p, x) for x in sub_p_list]
        if os.path.isfile(sub_p) and sub_p.endswith('.hdf5'):
            print(sub_p, dest)
            resave_hdf5(sub_p, dest)
            verify_structure(sub_p, dest)
        files = files[1:]
        gc.collect();

def check_fun(name, node):
    if isinstance(node, h5py.Dataset):
        tmp = node[:]
        dtypes_set.add(tmp.dtype)
        if tmp.dtype in (np.int64, np.int32):
            tmp2 = tmp.astype(np.int16)
            err_dict[name] = np.linalg.norm(tmp - tmp2, 1)
        if tmp.dtype in (np.float64,):
            tmp2 = tmp.astype(np.float32)
            err_dict[name] = np.linalg.norm(tmp - tmp2, 1)
            
def get_group_attr_keys(buf, name, node):
    buf.append((name, tuple(node.attrs.keys())))

def resave_function(fd, name, node):
    if isinstance(node, h5py.Dataset):
        dataset_name, group_name = name[::-1].split('/', 1)
        group_name, dataset_name = group_name[::-1], dataset_name[::-1]
        branch = group_name.split('/')
        cur_group_name = ''
        group_list = [fd]
        for i, leaf in enumerate(branch):
            #cur_group_name = os.path.join(cur_group_name, branch[i])
            #if cur_group_name in current_group:
            #    current_group = current_group[cur_group_name]
            #else:
            #    current_group = current_group.create_group(cur_group_name)
            if leaf in group_list[-1]:
                current_group = group_list[-1][leaf]
            else:
                current_group = group_list[-1].create_group(leaf)
            group_list.append(current_group)
        tmp = node[:]
        if tmp.dtype in (np.int64, np.int32):
            tmp = tmp.astype(np.int16)
        if tmp.dtype in (np.float64,):
            tmp = tmp.astype(np.float32)
        print(name, group_name, branch)
        current_group = fd[group_name]
        current_group.create_dataset(
            name=dataset_name,
            data=tmp,
            compression="gzip",
            compression_opts=9,
            chunks=True,
            shuffle=True,
            fletcher32=True,
            #maxshape=(None, ) + X.shape[1:]
        )
        del tmp;
    elif isinstance(node, h5py.Group):
        pass
        #if name in fd:
        #    current_group = fd[name]
        #else:
        #    current_group = fd.create_group(name)
        
def resave_attrs(fd, name, node):
    for key, val in node.attrs.items():
        fd[name].attrs.create(
            key, data=val
        )
        #print(f"    {key}: {val}")
        

def resave_hdf5(path1, path2):
    with h5py.File(path2, 'a') as dest_fd:
        current_saver = functools.partial(resave_function, dest_fd)
        current_saver_attrs = functools.partial(resave_attrs, dest_fd)
        with h5py.File(path1, 'r') as source_fd:
            source_fd.visititems(current_saver)
            source_fd.visititems(current_saver_attrs)
            
def verify_structure(path1, path2):
    buf1, buf2 = [], []
    with h5py.File(path1, 'r') as fd:
        current_func = functools.partial(get_group_attr_keys, buf1)
        fd.visititems(current_func)
    with h5py.File(path2, 'r') as fd:
        current_func = functools.partial(get_group_attr_keys, buf2)
        fd.visititems(current_func)
    assert buf1 == buf2
    
def check_values_function(fd, buf, name, node):
    if not isinstance(node, h5py.Dataset):
        return
    #dataset_name, group_name = name[::-1].split('/', 1)
    #group_name, dataset_name = group_name[::-1], dataset_name[::-1]
    tmp = node[:]
    tmp2 = fd[name][:]
    err_nrm = np.linalg.norm(tmp - tmp2) / tmp.size
    buf.append((name, err_nrm))
    del tmp, tmp2;
    gc.collect();

    
def verify_values(path1, path2):
    buf = []
    with h5py.File(path1, 'r') as fd1:
        with h5py.File(path2, 'r') as fd2:
            current_func = functools.partial(check_values_function, fd2, buf)
            fd1.visititems(current_func)
    return buf

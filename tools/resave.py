# Tools for compressing the hdf5 files
# should be used after any cleaning procedures

import h5py
import numpy as np
import functools
import os
import gc

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
        



if __name__ == '__main__':
    source_path = './zenodo'
    dest_path = './zenodo_compressed'
    resav_dir(source_path, dest_path)


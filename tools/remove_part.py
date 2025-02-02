# Tool for removing internal parts of hdf5 files
# used specifically to remove si2 (2nd order Sobol indices)
# from the third experimental series results

import os
import gc
import h5py
import functools


def get_remove_dataset_parts(buf, removing_dataset_name, name, node):
    if isinstance(node, h5py.Dataset):
        dataset_name, group_name = name[::-1].split('/', 1)
        group_name, dataset_name = group_name[::-1], dataset_name[::-1]
        if dataset_name != removing_dataset_name:
            return
        buf.append(name)

def clean_dataset_part(path, removing_dataset_name):
    buf = []
    current_func = functools.partial(
        get_remove_dataset_parts, buf, removing_dataset_name
    )
    with h5py.File(path, 'a') as fd:
        fd.visititems(current_func)
        for p in buf:
            del fd[p]
                   

if __name__ == "__main__":
    dirpath = "./zenodo/3"
    part_to_remove = "si2"
    files = os.listdir(dirpath)
    while len(files) > 0:
        p = files.pop()
        sub_p = os.path.join(dirpath, p)
        if os.path.isdir(sub_p):
            sub_p_list = os.listdir(sub_p)
            files += [os.path.join(p, x) for x in sub_p_list]
        if os.path.isfile(sub_p) and sub_p.endswith('.hdf5'):
            print(sub_p)
            clean_dataset_part(sub_p, part_to_remove)
        gc.collect();
        

# Tool to validate the compression results
# Compares npz, pkl and hdf5 files in referencedirectory and
# one with compressed results

import sys
sys.path.append("../src/")
import os

import numpy as np
import h5py

import hdf5_utils
import data_loader.utils


def extract_leaves(path, filter_exts=None, allow_exts=None):
    if filter_exts is None:
        filter_exts = []
    list1 = os.listdir(path)
    list1 = [os.path.join(path, s) for s in list1]
    list1.sort()
    rv = []
    while len(list1) > 0:
        cur_s = list1.pop(0)
        if os.path.isfile(cur_s):
            _, cur_ext = os.path.splitext(cur_s)
            cur_ext = cur_ext.replace('.', '').lower()
            if cur_ext in filter_exts:
                continue
            if (allow_exts is not None) and (cur_ext not in allow_exts):
                continue
            rv.append(cur_s)
            continue
        sub_s = os.listdir(cur_s)
        list1 += [os.path.join(cur_s, x) for x in sub_s]
    return rv

def compare_hdf5_files(ref_path, chk_path):
    rv = []
    values = hdf5_utils.verify_values(ref_path, chk_path)
    return values

def check_npndarrays(a, b):
    try:
        rv = (a == b).all()
    except:
        rv = False
    if rv:
        return rv
    try:
        err = np.linalg.norm(a - b) / a.size
        rv = np.isclose(err, 0)
    except:
        rv = False
    return rv

def compare_dicts(ref_dict, chk_dict):
    problem_keys = []
    for key in ref_dict:
        d1 = ref_dict[key]
        d2 = chk_dict[key]
        if d1 is None:
            if d2 is None:
                continue
            problem_keys.append(key)
        elif isinstance(d1, np.ndarray):
            if not check_npndarrays(d1, d2):
                problem_keys.append(key)
        else:
            problem_keys.append(key)
    return problem_keys

def compare_pkl_files(ref_path, chk_path):
    '''
    list, or list of dicts (key, np.ndarray)
    '''
    d1 = data_loader.utils.load_pickled_data(ref_path)
    d2 = data_loader.utils.load_pickled_data(ref_path)
    problems = []
    if isinstance(d1, list):
        if isinstance(d1[0], dict):
            for i in range(len(d1)):
                cur_dict1, cur_dict2 = d1[i], d2[i]
                cur_problems = compare_dicts(cur_dict1, cur_dict2)
                if len(cur_problems) > 0:
                    problems.append((i, cur_problems))
        else:
            if not check_npndarrays(np.array(d1), np.array(d2)):
                problems += ['non-matched content']
    else:
        problems += ['non-matched content']
    return problems

def compare_npz_files(ref_path, chk_path):
    d1 = np.load(ref_path, allow_pickle=True)
    d2 = np.load(chk_path, allow_pickle=True)
    d1 = dict((k, v) for k, v in d1.items())
    d2 = dict((k, v) for k, v in d2.items())
    problems = compare_dicts(d1, d2)
    return problems

_compare_func_dict = {
    'hdf5': compare_hdf5_files,
    'pkl': compare_pkl_files,
    'npz': compare_npz_files,
}


def compare_results(reference_directory_path, results_directory_path):
    file_list = extract_leaves(
        reference_directory_path,
        allow_exts=_compare_func_dict.keys(),
    )
    problems = []
    for p1 in file_list:
        rel_path = os.path.relpath(p1, reference_directory_path)
        p2 = os.path.join(results_directory_path, rel_path)
        _, cur_ext = os.path.splitext(p1)
        cur_ext = cur_ext.replace('.', '').lower()
        # no match -- case in py3.8
        try:
            cur_output = _compare_func_dict[cur_ext](p1, p2)
            if len(cur_output) > 0:
                problems.append((rel_path, cur_output))
            print(rel_path, cur_output)
        except:
            problems.append((rel_path, 'could not compare files'))
            print(rel_path, ': could not compare files')
    return problems

if __name__ == "__main__":
    reference_directory_path = "./results_compressed/"
    results_directory_path = "./results/"
    compare_results(reference_directory_path, results_directory_path)


import importlib
import random
import torch
import types
import sys

import multiprocessing

def torch_init():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def torch_seeding(torch_seed, random_seed=None):
    if random_seed is None:
        random_seed = torch_seed
    # setting manual seeds
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed) # multi GPU case
    random.seed(random_seed)  # Python random module.
    torch.manual_seed(torch_seed) # once again!..
    
def module_reloader(module_names):
    for name in module_names:
        m = importlib.import_module(name)
        importlib.reload(m)
        
def setup_multiprocessing():
    multiprocessing.set_start_method("spawn");
        
# https://stackoverflow.com/questions/58597680/how-can-a-python-decorator-change-calls-in-decorated-function
#def update_function_context(func, upd_dict):
#    globals = func.__globals__.copy()
#    globals.update(upd_dict)
#    new_func = types.FunctionType(
#        func.__code__,
#        globals,
#        name=func.__name__,
#        argdefs=func.__defaults__,
#        closure=func.__closure__
#    )
#    #new_func.__dict__.update(func.__dict__)
#    #new_func.__dict__.update(upd_dict)
#    return new_func
#
#def override_functions_in_module(module_name, func_dict):
#    module = sys.modules.get(module_name)
#    for key, value in func_dict.items():
#        setattr(module, key, value)
#        globals()[key] = value
#    #return update_function_context(func, func_dict)
# salib! ZeroDivision!
    
import functools
import torch
import torchvision



def vmap(function):
    def fun(X, **params):
        return torch.stack([function(x, **params) for x in X])
    return fun

def register_forward_hook_by_name(model, hook_function, desired_module_names):
    handlers = {}
    for name, module in model.named_modules():
        if name in desired_module_names:
            # This adds global state to the nn.module module and it is only intended for debugging/profiling purposes.
            handler = module.register_forward_hook(functools.partial(hook_function, name))
            #handler = module.register_forward_hook(hook_function)
            #handler = module.register_forward_hook(functools.partial(hook_function, module))
            handlers[name] = handler
    return handlers

def register_backward_hook_by_name(model, hook_function, desired_module_names):
    handlers = {}
    for name, module in model.named_modules():
        if name in desired_module_names:
            handler = module.register_full_backward_hook(
                #hook_function
                functools.partial(hook_function, name)
                #functools.partial(hook_function, module)
            )
            handlers[name] = handler
    return handlers

def remove_all_hooks_by_dict(handlers):
    for name, handler in handlers.items():
        handler.remove()
        
def single_random_apply_short(construct, p):
    return torchvision.transforms.RandomApply(
        torch.nn.ModuleList([construct]), p
    )

# !!NOTE: dim=1 means that the input should be 4D!!
def img_normalize_old(image, eps=1e-20, unit=False):
    '''
    Attention! Experiments were done using this function.
    The main drawback is that it works only with one spatial axis
    (dim=1, because the input is meant to be 3-dimensional of (channel, spat_1, spat_2) shape.
    
    Below is modified version that deal with both spatial dimensions,
    what should lead to more precise results.
    '''
    image = image - image.min(dim=1, keepdim=True).values
    image = image / (image.max(dim=1, keepdim=True).values + eps)
    if unit:
        return image
    return (image - 0.5) / 0.5


#def img_normalize(image, eps=1e-20, unit=False):
def img_normalize(image, eps=1e-6, unit=False):
    #shape = image.shape
    #new_shape = shape[:-2] + (shape[-2]*shape[-1], )
    #image = image - image.min(dim=1, keepdim=True).values
    #dim = (-2, -1)
    min_imag = image.min(dim=-1, keepdim=True).values
    min_imag = min_imag.min(dim=-2, keepdim=True).values
    #image = image - image.min(dim=dim, keepdim=True).values
    #max_imag = image.max(dim=dim, keepdim=True).values
    image = image - min_imag
    max_imag = image.max(dim=-1, keepdim=True).values
    max_imag = max_imag.max(dim=-2, keepdim=True).values
    #max_imag = image.max(dim=dim, keepdim=True).values
    max_imag[max_imag < eps] = 1.
    image = image / max_imag
    if unit:
        return image
    return (image - 0.5) / 0.5
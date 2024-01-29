# nothing to import!

def is_leaf(module):
    try:
        next(module.children())
    except StopIteration:
        return True
    return False

def get_flatten_leaves(model):
    leaves = []
    for name, module in model.named_modules():
        if is_leaf(module):
            leaves.append(name)
    return leaves

def remove_leaves_by_substr(network_leaves, substr_list):
    output = network_leaves
    for substr in substr_list:
        output = list(filter(lambda x: not x.endswith(substr), output))
    return output

def feedforward_model_subinference(model, input_X, terminal_leaf_name):
    output_y = input_X
    for name, module in model.named_modules():
        if is_leaf(module):
            output_y = module(output_y)
            if name == terminal_leaf_name:
                break
    return output_y

def access_layer(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            break
    return module
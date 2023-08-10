import pickle
import sys
import collections
import torch
import zipfile
from collections import OrderedDict
import json
import time

from phantom_model import Transformer as PhantomTransformer, ModelArgs as PhantomModelArgs


# use ones from llama2.c
sys.path.insert(0, '../llama2.c')
from model import Transformer, ModelArgs

# sys.argv[1] - path to consolidated.00.pth
# sys.argv[2] - path to params.json

def load_torch_pickle(file):
    class DummyObj:
        def __init__(self, *args):
            pass

    class DummyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            #print(f'custom_find_class {module} {name}')
            if name == 'OrderedDict':
                return collections.OrderedDict
            return DummyObj

    def persistent_load(saved_id):
        #print(f'persistent_load {saved_id}')
        typename, storage_type, key, location, numel = saved_id
        return None
        
    unpickler = DummyUnpickler(file)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    return list(result.keys())


def load_state_dict():
    manual_state_dict = OrderedDict()

    with zipfile.ZipFile(sys.argv[1]) as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            module_list = load_torch_pickle(pickle_file)
        for i, module_name in enumerate(module_list):
            with checkpoint_zip.open(f'consolidated/data/{i}', 'r') as data_file:
                buf = data_file.read()
            manual_state_dict[module_name] = torch.tensor(torch.UntypedStorage.from_buffer(buf, dtype=torch.bfloat16, byte_order='little'), dtype=torch.bfloat16)
    return manual_state_dict

def load_llama7b():
    vocab_size = 32000
    with open(sys.argv[2], 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    model_conf = ModelArgs(**config)
    model = Transformer(model_conf)
    return model

def load_phantom_llama7b():
    vocab_size = 32000
    with open(sys.argv[2], 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    model_conf = PhantomModelArgs(**config)
    model = PhantomTransformer(model_conf)
    return model

## loading with torch (default mode)
def llama7b_torch():
    model = load_llama7b()

    state_dict = torch.load(sys.argv[1])
    
    model.load_state_dict(state_dict, strict=False)
    print(model)
    return model


## loading manually
def llama7b_manual():
    print('loading state dict')
    state_dict = load_state_dict()
    print(f'loaded state_dict with {len(state_dict)} entries')
    print('loading model')
    model = load_llama7b()

    def fix_shapes_rec(module, prefix=''):
        nonlocal state_dict
        if hasattr(module, 'weight'):
            key = f'{prefix}weight'
            loaded_shape = state_dict[key].shape
            print(f'found {key} with shapes {module.weight.shape} and {loaded_shape}.')
            state_dict[key] = state_dict[key].reshape(module.weight.shape)

        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                fix_shapes_rec(child, child_prefix)

    fix_shapes_rec(model)

    model.load_state_dict(state_dict, strict=False)
    print(model)
    return model

def load_phantom_module_list():
    with zipfile.ZipFile(sys.argv[1]) as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            return load_torch_pickle(pickle_file)

# module subset is a list of tuples ('submodule.path', id_in_original_state_dict)
def populate_phantom_state_dict(module_subset):
    state_dict = OrderedDict()

    with zipfile.ZipFile(sys.argv[1]) as checkpoint_zip:
        for i, module_name in module_subset:
            with checkpoint_zip.open(f'consolidated/data/{i}', 'r') as data_file:
                buf = data_file.read()
            state_dict[module_name] = torch.tensor(torch.UntypedStorage.from_buffer(buf, dtype=torch.bfloat16, byte_order='little'), dtype=torch.bfloat16)
    return state_dict

def llama7b_phantom():
    modules = load_phantom_module_list()
    model = load_phantom_llama7b()

    # first manually populate layers one by one
    for i, phantom_layer in enumerate(model.layers):
        prefix = f'layers.{i}.'
        relevant_modules = [(j, k[len(prefix):]) for j, k in enumerate(modules) if k.startswith(prefix)]
        print(f'{i} {relevant_modules}')
        
        ## we need to 
        # a. load these into dictionary
        state_dict = populate_phantom_state_dict(relevant_modules)
        print(state_dict)
        # b. fix shapes manually
        # c. load_state_dict
        phantom_layer.populate_state_dict(state_dict)
        
    # now we need to populate everything except layers using strict=False
    remaining_modules = [(j, k) for j, k in enumerate(modules) if not k.startswith('layers.')]
    print(remaining_modules)

    state_dict = populate_phantom_state_dict(remaining_modules)
    def fix_shapes_rec(module, prefix=''):
        nonlocal state_dict
        if hasattr(module, 'weight'):
            key = f'{prefix}weight'
            if key in state_dict.keys():
                loaded_shape = state_dict[key].shape
                print(f'found {key} with shapes {module.weight.shape} and {loaded_shape}.')
                state_dict[key] = state_dict[key].reshape(module.weight.shape)
            else:
                print(f'not found {key}')

        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                fix_shapes_rec(child, child_prefix)

    fix_shapes_rec(model)
    model.load_state_dict(state_dict, strict=False)
    print(model)
    return model


X = torch.arange(500).view(1, 500).to('cpu')

start = time.time()
model = llama7b_phantom().to('cpu')
print(f'loaded model in {time.time() - start} seconds')
start = time.time()
print(model(X))
print(f'evaluated model in {time.time() - start} seconds')


"""
This produces RuntimeError: MPS backend out of memory 
X = torch.arange(500).view(1, 500).to('mps')
model = llama7b_torch().to('mps')
print(model(X))
"""

X = torch.arange(500).view(1, 500).to('cpu')

start = time.time()
model = llama7b_torch().to('cpu')
print(f'loaded model in {time.time() - start} seconds')

"""
start = time.time()
print(model(X))
print(f'evaluated model in {time.time() - start} seconds')
"""

# Output from Apple M2 @ 24Gb RAM running on CPU:
"""
loaded model in 93.34452700614929 seconds
tensor([[[-7.7749,  0.2068,  3.7988,  ..., -2.6116, -3.5703, -1.7604]]],
       grad_fn=<UnsafeViewBackward0>)
evaluated model in 132.63551712036133 seconds

"""

# try same with no_grad:
with torch.no_grad():
    start = time.time()
    print(model(X))
    print(f'evaluated model in {time.time() - start} seconds')
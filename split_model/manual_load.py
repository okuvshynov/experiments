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

def prepare_llama_config():
    vocab_size = 32000
    with open(sys.argv[2], 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    return config

def load_llama7b():

    model_conf = ModelArgs(**prepare_llama_config())
    model = Transformer(model_conf)
    return model

def load_phantom_llama7b():
    model_conf = PhantomModelArgs(**prepare_llama_config())
    model = PhantomTransformer(model_conf)
    return model

## loading with torch (default mode)
def llama7b_torch():
    model = load_llama7b()

    state_dict = torch.load(sys.argv[1])
    
    model.load_state_dict(state_dict, strict=False)
    #print(model)
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
        #print(f'{i} {relevant_modules}')
        
        ## we need to 
        # a. load these into dictionary
        state_dict = populate_phantom_state_dict(relevant_modules)
        #print(state_dict)
        # b. fix shapes manually
        # c. load_state_dict
        phantom_layer.populate_state_dict(state_dict)
        
    # now we need to populate everything except layers using strict=False
    remaining_modules = [(j, k) for j, k in enumerate(modules) if not k.startswith('layers.')]
    #print(remaining_modules)

    state_dict = populate_phantom_state_dict(remaining_modules)
    def fix_shapes_rec(module, prefix=''):
        nonlocal state_dict
        if hasattr(module, 'weight'):
            key = f'{prefix}weight'
            if key in state_dict.keys():
                loaded_shape = state_dict[key].shape
                #print(f'found {key} with shapes {module.weight.shape} and {loaded_shape}.')
                state_dict[key] = state_dict[key].reshape(module.weight.shape)
            else:
                print(f'not found {key}')

        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                fix_shapes_rec(child, child_prefix)

    fix_shapes_rec(model)
    model.load_state_dict(state_dict, strict=False)
    #print(model)
    return model

device = 'mps'

batch_size = 1
X = torch.arange(500 * batch_size).view(batch_size, 500).to(device)

start = time.time()
model = llama7b_phantom().to(device)
print(f'loaded phantom model in {time.time() - start} seconds')
start = time.time()
phantom_y = model(X)
torch.set_printoptions(profile="full", precision=10)
print(phantom_y)
print(f'evaluated phantom model in {time.time() - start} seconds')


"""
trying to put whole model to mps produces RuntimeError: MPS backend out of memory:
model = llama7b_torch().to('mps')
print(model(X))
"""
X = X.to('cpu')
start = time.time()
model = llama7b_torch().to('cpu')
print(f'loaded model in {time.time() - start} seconds')
start = time.time()
default_y = model(X)
print(default_y)
print(f'evaluated model in {time.time() - start} seconds')

same_y = torch.allclose(phantom_y.cpu(), default_y.cpu())

print(f'output for default torch model and phantom model is same: {same_y}')


# Output from Apple M2 @ 24Gb RAM running on MPS and CPU:
"""
% python split_model/manual_load.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
loaded phantom model in 88.39778780937195 seconds
tensor([[[-7.7749,  0.2068,  3.7988,  ..., -2.6116, -3.5703, -1.7604]]],
       device='mps:0', grad_fn=<LinearBackward0>)
evaluated phantom model in 13.354475975036621 seconds
loaded model in 93.51274299621582 seconds
tensor([[[-7.7749,  0.2068,  3.7988,  ..., -2.6116, -3.5703, -1.7604]]],
       grad_fn=<UnsafeViewBackward0>)
evaluated model in 129.28788685798645 seconds
output for default torch model and phantom model is same: False

"""

# Looks like the output is not exactly the same but very close - 
# I wonder if it's device/precision thing. 
# Let's check with running both on CPU  
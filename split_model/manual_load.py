import pickle
import sys
import collections
import torch
import zipfile
from collections import OrderedDict
import json
import time
import os

from phantom_model import Transformer as PhantomTransformer, ModelArgs as PhantomModelArgs

llama2_7b_path = sys.argv[1]
params_path = os.path.join(llama2_7b_path, "params.json")
weights_path = os.path.join(llama2_7b_path, "consolidated.00.pth")

# use ones from llama2.c
sys.path.insert(0, '../llama2.c')
from model import Transformer, ModelArgs

def load_torch_pickle(file):
    class DummyObj:
        def __init__(self, *args):
            pass

    class DummyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'OrderedDict':
                return collections.OrderedDict
            return DummyObj
        
    unpickler = DummyUnpickler(file)
    unpickler.persistent_load = lambda _: None
    result = unpickler.load()

    return list(result.keys())

def prepare_llama_config():
    vocab_size = 32000
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    return config

## loading with torch (default mode)
def llama7b_torch():
    model_conf = ModelArgs(**prepare_llama_config())
    model = Transformer(model_conf)
    state_dict = torch.load(weights_path)    
    model.load_state_dict(state_dict, strict=False)
    return model

def load_phantom_module_list():
    with zipfile.ZipFile(weights_path) as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            return load_torch_pickle(pickle_file)

# module subset is a list of tuples ('submodule.path', id_in_original_state_dict)
def populate_phantom_state_dict(module_subset):
    state_dict = OrderedDict()

    with zipfile.ZipFile(weights_path) as checkpoint_zip:
        for i, module_name in module_subset:
            with checkpoint_zip.open(f'consolidated/data/{i}', 'r') as data_file:
                buf = data_file.read()
            state_dict[module_name] = torch.tensor(torch.UntypedStorage.from_buffer(buf, dtype=torch.bfloat16, byte_order='little'), dtype=torch.bfloat16)
    return state_dict

def llama7b_phantom():
    modules = load_phantom_module_list()
    print(f'Loaded {len(modules)} module metadata')
    model_conf = PhantomModelArgs(**prepare_llama_config())
    model = PhantomTransformer(model_conf)
    print('Created blank model')

    # first manually populate layers one by one
    for i, phantom_layer in enumerate(model.layers):
        print(f'processing transformer block {i}')
        prefix = f'layers.{i}.'
        relevant_modules = [(j, k[len(prefix):]) for j, k in enumerate(modules) if k.startswith(prefix)]

        # a. load these into dictionary
        state_dict = populate_phantom_state_dict(relevant_modules)
        
        # b. fix shapes manually
        # c. load_state_dict
        phantom_layer.populate_state_dict(state_dict)
        
    # now we need to populate everything except transformer block layers with strict=False
    remaining_modules = [(j, k) for j, k in enumerate(modules) if not k.startswith('layers.')]
    state_dict = populate_phantom_state_dict(remaining_modules)
    def fix_shapes_rec(module, prefix=''):
        nonlocal state_dict
        if hasattr(module, 'weight'):
            key = f'{prefix}weight'
            state_dict[key] = state_dict[key].reshape(module.weight.shape)

        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                fix_shapes_rec(child, child_prefix)

    fix_shapes_rec(model)
    model.load_state_dict(state_dict, strict=False)
    print('populated all weights to model')

    return model

device = 'cpu'
batch_size = 1
X = torch.arange(500 * batch_size).view(batch_size, 500).to(device)

## Running phantom
start = time.time()
model = llama7b_phantom().to(device)
print(f'loaded phantom model in {time.time() - start} seconds')
start = time.time()
phantom_y = model(X)
print(phantom_y)
print(f'evaluated phantom model on batch_size={batch_size} in {time.time() - start} seconds')

## Running default
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


"""
Output from Apple M2 @ 24Gb RAM running on MPS and CPU:

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


Looks like the output is not exactly the same but very close - 
I wonder if it's device/precision thing. 
Let's check with running both on CPU  

When both running on CPU we got match.

...
evaluated model in 135.04086709022522 seconds
output for default torch model and phantom model is same: True
"""

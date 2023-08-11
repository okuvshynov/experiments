import collections
import json
import os
import pickle
import sys
import time
import zipfile
from collections import OrderedDict

import torch

from phantom_model import Transformer as PhantomTransformer, ModelArgs as PhantomModelArgs

llama2_7b_path = sys.argv[1]
params_path = os.path.join(llama2_7b_path, "params.json")
weights_path = os.path.join(llama2_7b_path, "consolidated.00.pth")

vocab_size = 32000

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
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    return config

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
length = 50

assert(length * batch_size < vocab_size)

X = torch.arange(length * batch_size).view(batch_size, length).to(device)
Y = X + 1
#print(X, Y)

start = time.time()
model = llama7b_phantom().to(device)
print(f'loaded model in {time.time() - start} seconds')
start = time.time()

# insane LR to see difference after 1 iteration with 1 sample
opt = torch.optim.SGD(model.parameters(), lr=100.0)

start = time.time()
logits = model(X, Y)
print(f'forward pass in {time.time() - start} seconds')
layer_13 = model.layers[13].loaded_inner()
print(layer_13.attention.wq.weight)

start = time.time()
opt.zero_grad()
loss = model.last_loss
loss.backward()
opt.step()
print(f'backward pass in {time.time() - start} seconds')

layer_13 = model.layers[13].loaded_inner()
print(layer_13.attention.wq.weight)
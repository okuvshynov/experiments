import pickle
import sys
import collections
import torch
import zipfile
from collections import OrderedDict
import numpy as np
import json

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
                # TODO: is it float16 of bfloat16?
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


X = torch.arange(500).view(1, 500)

model = llama7b_torch()

print(model(X))
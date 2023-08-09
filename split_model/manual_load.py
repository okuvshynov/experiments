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
            print(f'custom_find_class {module} {name}')
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
        print(checkpoint_zip.namelist())
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            module_list = load_torch_pickle(pickle_file)
        for i, module_name in enumerate(module_list):
            print(i, module_name)
            with checkpoint_zip.open(f'consolidated/data/{i}', 'r') as data_file:
                # TODO: is it float16 of bfloat16?
                data = np.frombuffer(data_file.read(), dtype=np.float16)
            manual_state_dict[module_name] = torch.tensor(data)
            print(module_name, manual_state_dict[module_name].shape)
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
    
    # model definition is a little different
    del state_dict['rope.freqs']
    
    model.load_state_dict(state_dict)
    print(model)


## loading manually
def llama7b_manual():
    model = load_llama7b()

    state_dict = load_state_dict()

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

    # model definition is a little different
    del state_dict['rope.freqs']

    model.load_state_dict(state_dict)
    print(model)

llama7b_manual()
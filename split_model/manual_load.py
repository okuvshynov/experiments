import pickle
import sys
import collections
import torch
import zipfile
from collections import OrderedDict
import numpy as np
import json

# use ones from llama2.c
from model import Transformer, ModelArgs

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


def load_manual_state_dict():
    manual_state_dict = OrderedDict()

    with zipfile.ZipFile(sys.argv[1]) as checkpoint_zip:
        print(checkpoint_zip.namelist())
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            module_list = load_torch_pickle(pickle_file)
        for i, module_name in enumerate(module_list):
            print(i, module_name)
            with checkpoint_zip.open(f'consolidated/data/{i}', 'r') as data_file:
                data = np.frombuffer(data_file.read(), dtype=np.float16)
            manual_state_dict[module_name] = torch.tensor(data)
            print(module_name, manual_state_dict[module_name].shape)


def torch_module_dfs(module, prefix=''):
    res = []
    if hasattr(module, 'weight'):
        res.append((f'{prefix}weight', module.weight.shape))
        print(f'{prefix}weight -- {module}')
    for name, child in module._modules.items():
        if child is not None:
            child_prefix = prefix + name + '.'
            res.extend(torch_module_dfs(child, child_prefix))
    return res

#

def load_llama7b_torch():
    vocab_size = 32000
    with open(sys.argv[2], 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    model_conf = ModelArgs(**config)
    model = Transformer(model_conf)
    return model

def run_llama7b_torch():
    torch_state_dict = torch.load(sys.argv[1])
    
    # model definition is a little different
    del torch_state_dict['rope.freqs']
    #print(f"rope.freqs from state_dict: {torch_state_dict['rope.freqs']}")
    model = load_llama7b_torch()
    model.load_state_dict(torch_state_dict)
    print(model)

run_llama7b_torch()
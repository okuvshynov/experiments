# This is partial load/save functionality for llama2 models.
# In contrast with model.py, which has model definition itself and can run 
# other model with different settings, this one is specifically focusing on 
# loading/saving the model as it is shared by Meta. 
# It relies on specific data types, etc.

import pickle
import sys
import collections
import torch
import zipfile
import json
import os
import ctypes

from phantom_model import Transformer as PhantomTransformer, ModelArgs as PhantomModelArgs

vocab_size = 32000

def prepare_llama_config(params_path):
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    config['dropout'] = 0.5
    return config

def load_torch_pickle(file):
    class DummyObj:
        def __init__(self, *args):
            pass

    class DummyUnpickler(pickle.Unpickler):
        def find_class(self, _module, name):
            if name == 'OrderedDict':
                return collections.OrderedDict
            return DummyObj
        
    unpickler = DummyUnpickler(file)
    unpickler.persistent_load = lambda _: None
    result = unpickler.load()

    return list(result.keys())

def load_phantom_module_list(weights_path):
    with zipfile.ZipFile(weights_path) as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            return load_torch_pickle(pickle_file)

# module subset is a list of tuples ('submodule.path', id_in_original_state_dict)
def populate_phantom_state_dict(module_subset, weights_path):
    state_dict = collections.OrderedDict()

    with zipfile.ZipFile(weights_path) as checkpoint_zip:
        for i, module_name in module_subset:
            with checkpoint_zip.open(f'consolidated/data/{i}', 'r') as data_file:
                buf = data_file.read()
            state_dict[module_name] = torch.tensor(torch.UntypedStorage.from_buffer(buf, dtype=torch.bfloat16, byte_order='little'), dtype=torch.bfloat16)
    return state_dict

def llama7b_phantom(llama2_7b_path):
    params_path = os.path.join(llama2_7b_path, "params.json")
    weights_path = os.path.join(llama2_7b_path, "consolidated.00.pth")

    modules = load_phantom_module_list(weights_path)
    print(f'Loaded {len(modules)} module metadata')
    model_conf = PhantomModelArgs(**prepare_llama_config(params_path))
    model = PhantomTransformer(model_conf)
    print('Created blank model')

    sys.stdout.write('processing transformer blocks ')
    # first manually populate layers one by one
    for i, phantom_layer in enumerate(model.layers):
        sys.stdout.write('.')
        sys.stdout.flush()
        prefix = f'layers.{i}.'
        relevant_modules = [(j, k[len(prefix):]) for j, k in enumerate(modules) if k.startswith(prefix)]

        # a. load these into dictionary
        state_dict = populate_phantom_state_dict(relevant_modules, weights_path)
        
        # b. fix shapes manually
        # c. load_state_dict
        phantom_layer.from_state_dict(state_dict)
    print(' DONE')
        
    # now we need to populate everything except transformer block layers with strict=False
    remaining_modules = [(j, k) for j, k in enumerate(modules) if not k.startswith('layers.')]
    state_dict = populate_phantom_state_dict(remaining_modules, weights_path)
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


def save_model(model, original_path, new_path):
    state_dict = model.state_dict()
    for i, phantom_layer in enumerate(model.layers):
        sys.stdout.write('.')
        sys.stdout.flush()
        prefix = f'layers.{i}.'
        module_state_dict = phantom_layer.to_state_dict()
        for k, t in module_state_dict.items():
            state_dict[prefix + k] = t

    # loading data.pkl from original file
    old_weights_path = os.path.join(original_path, "consolidated.00.pth")
    modules = load_phantom_module_list(old_weights_path)

    with zipfile.ZipFile(old_weights_path) as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            data_pkl = pickle_file.read()

    new_weights_path = os.path.join(new_path, "consolidated.00.pth")
    
    with zipfile.ZipFile(new_weights_path, mode='w') as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'w') as pickle_file:
            pickle_file.write(data_pkl)
        for i, key in enumerate(modules):
            if key not in state_dict.keys():
                continue
            # prepare buffer of the right type from the tensor
            t = state_dict[key].to(torch.bfloat16)
            
            size = t.untyped_storage().nbytes()
            addr = t.untyped_storage().data_ptr()
            print(f'{t.dtype} -- {size}')

            buffer = bytes((ctypes.c_char * size).from_address(addr))

            with checkpoint_zip.open(f'consolidated/data/{i}', 'w') as data_file:
                data_file.write(buffer)
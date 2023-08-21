# This is partial load/save functionality for llama2 models.
# In contrast with model.py, which has model definition itself and can run 
# other model with different settings, this one is specifically focusing on 
# loading/saving the model as it is shared by Meta. 
# It relies on specific data types, file names, etc..

import pickle
import collections
import torch
import zipfile
import json
import os
import ctypes
import gc
import shutil

from blackbox_model import Transformer, ModelArgs
from utils import peak_rss_mb

vocab_size = 32000

def _prepare_llama_config(params_path, **kwargs):
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    for k, v in kwargs.items():
        config[k] = v
    return config

def _load_torch_pickle(file):
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

def _load_module_list(weights_path):
    with zipfile.ZipFile(weights_path) as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            return _load_torch_pickle(pickle_file)

# module subset is a list of tuples ('submodule.path', id_in_original_state_dict)
def _populate_state_dict(module_subset, weights_path):
    state_dict = collections.OrderedDict()

    with zipfile.ZipFile(weights_path) as checkpoint_zip:
        names = checkpoint_zip.namelist()
        for i, module_name in module_subset:
            name = f'consolidated/data/{i}'
            if not name in names:
                print(f'{module_name}[{name}] is missing in the archive')
                continue
            with checkpoint_zip.open(name, 'r') as data_file:
                buf = data_file.read()
            state_dict[module_name] = torch.tensor(torch.UntypedStorage.from_buffer(buf, dtype=torch.bfloat16, byte_order='little'), dtype=torch.bfloat16)
    return state_dict

def load_llama7b(llama2_7b_path, **kwargs):
    params_path = os.path.join(llama2_7b_path, "params.json")
    weights_path = os.path.join(llama2_7b_path, "consolidated.00.pth")

    modules = _load_module_list(weights_path)
    print(f'Loaded {len(modules)} module metadata')
    model_conf = ModelArgs(**_prepare_llama_config(params_path, **kwargs))
    model = Transformer(model_conf)
    print(f'peak rss: {peak_rss_mb()}')

    # first manually populate layers one by one
    for i, layer in enumerate(model.layers):
        prefix = f'layers.{i}.'
        relevant_modules = [(j, k[len(prefix):]) for j, k in enumerate(modules) if k.startswith(prefix)]

        state_dict = _populate_state_dict(relevant_modules, weights_path)
        
        layer.from_state_dict(state_dict)
        gc.collect()
        print(f'peak rss after loading layer {i}: {peak_rss_mb()}')

    prefix = 'output.'
    output_module = [(j, k[len(prefix):]) for j, k in enumerate(modules) if k.startswith(prefix)]
    model.output.from_state_dict(_populate_state_dict(output_module, weights_path))
    gc.collect()
    print(f'peak rss after output: {peak_rss_mb()}')
    
    # embed
    prefix = 'tok_embeddings.'
    embed_module = [(j, k[len(prefix):]) for j, k in enumerate(modules) if k.startswith(prefix)]
    model.tok_embeddings.from_state_dict(_populate_state_dict(embed_module, weights_path))
    gc.collect()
    print(f'peak rss after embeddings: {peak_rss_mb()}')

    remaining_modules = [(j, k) for j, k in enumerate(modules) if (not (k.startswith('layers.') or (k.startswith('output.')) or (k.startswith('tok_embeddings.')))) ]
    state_dict = _populate_state_dict(remaining_modules, weights_path)
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
    print(f'loading rest of modules: {model.load_state_dict(state_dict, strict=False)}')
    print('populated all weights to model')

    return model


def save_llama7b(model, original_path, new_path):
    state_dict = model.state_dict()

    # populate back the modules we have offloaded to the normal paths
    for i, layer in enumerate(model.layers):
        for k, t in layer.to_state_dict().items():
            state_dict[f'layers.{i}.{k}'] = t

    for k, t in model.output.to_state_dict().items():
        state_dict[f'output.{k}'] = t
    for k, t in model.tok_embeddings.to_state_dict().items():
        state_dict[f'tok_embeddings.{k}'] = t

    # loading data.pkl from original file
    old_weights_path = os.path.join(original_path, "consolidated.00.pth")
    modules = _load_module_list(old_weights_path)


    with zipfile.ZipFile(old_weights_path) as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            data_pkl = pickle_file.read()
        # copy rope.freqs which we are not using
        with checkpoint_zip.open('consolidated/data.pkl', 'r') as pickle_file:
            data_pkl = pickle_file.read()

    os.makedirs(new_path, exist_ok=True)
    new_weights_path = os.path.join(new_path, "consolidated.00.pth")
    shutil.copy2(os.path.join(original_path, "params.json"), new_path)
    
    with zipfile.ZipFile(new_weights_path, mode='w') as checkpoint_zip:
        with checkpoint_zip.open('consolidated/data.pkl', 'w') as pickle_file:
            pickle_file.write(data_pkl)
        for i, key in enumerate(modules):
            if key not in state_dict.keys():
                continue
            # prepare buffer of the right type from the tensor
            t = state_dict[key].cpu().to(torch.bfloat16)
            
            size = t.untyped_storage().nbytes()
            addr = t.untyped_storage().data_ptr()
            buffer = bytes((ctypes.c_char * size).from_address(addr))

            with checkpoint_zip.open(f'consolidated/data/{i}', 'w') as data_file:
                data_file.write(buffer)
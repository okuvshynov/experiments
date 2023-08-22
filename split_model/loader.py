import torch
import os
import json
import gc
import sys
import shutil

from blackbox_model import Transformer, ModelArgs
from utils import peak_rss_mb

vocab_size = 32000

def load_llama2_7b(path, **kwargs):
    weights_path = os.path.join(path, 'consolidated.00.pth')
    params_path = os.path.join(path, 'params.json')
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())

    config['vocab_size'] = vocab_size
    for k, v in kwargs.items():
        config[k] = v

    model = Transformer(ModelArgs(**config))
    checkpoint = torch.load(weights_path)

    for i, layer in enumerate(model.layers):
        prefix = f'layers.{i}.'
        state_dict = {k[len(prefix):]: w for k, w in checkpoint.items() if k.startswith(prefix)}
        layer.from_state_dict(state_dict, fix_shapes=False)

        # this might be not particularly important in a 'fit into memory' sense - 
        # only peak memory matters. However, in practice that might mean 'need to use swap less'
        # to make loading a little faster.
        for k in state_dict.keys():
            del checkpoint[f'{prefix}{k}']
        gc.collect()
        print(f'peak rss after loading layer {i}: {peak_rss_mb()}')

    model.output.from_state_dict({'weight': checkpoint['output.weight']}, fix_shapes=False)
    del checkpoint['output.weight']
    gc.collect()
    print(f'peak rss after output: {peak_rss_mb()}')

    model.tok_embeddings.from_state_dict({'weight': checkpoint['tok_embeddings.weight']}, fix_shapes=False)
    del checkpoint['tok_embeddings.weight']
    gc.collect()
    print(f'peak rss after tok_embeddings: {peak_rss_mb()}')

    model.load_state_dict(checkpoint, strict=False)
    return model

def save_llama2_7b(model, new_path, original_path):
    state_dict = model.state_dict()
    for i, layer in enumerate(model.layers):
        for k, t in layer.to_state_dict().items():
            state_dict[f'layers.{i}.{k}'] = t
    for k, t in model.output.to_state_dict().items():
        state_dict[f'output.{k}'] = t
    for k, t in model.tok_embeddings.to_state_dict().items():
        state_dict[f'tok_embeddings.{k}'] = t

    os.makedirs(new_path, exist_ok=True)
    weights_path = os.path.join(new_path, 'consolidated.00.pth')
    torch.save(state_dict, weights_path)
    shutil.copy2(os.path.join(original_path, "params.json"), new_path)

if __name__ == '__main__':
    load_llama2_7b(sys.argv[1])
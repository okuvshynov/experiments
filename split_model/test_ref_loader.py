import sys
import torch
import json
import os

# use ones from llama2.c
sys.path.insert(0, '../llama2.c')
from model import Transformer, ModelArgs

vocab_size = 32000

def prepare_llama_config(params_path, **kwargs):
    with open(params_path, 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    for k, v in kwargs.items():
        config[k] = v
    return config

## loading with torch (default mode)
def llama7b_torch(llama2_7b_path, **kwargs):
    params_path = os.path.join(llama2_7b_path, "params.json")
    weights_path = os.path.join(llama2_7b_path, "consolidated.00.pth")
    model_conf = ModelArgs(**prepare_llama_config(params_path, **kwargs))
    model = Transformer(model_conf)
    state_dict = torch.load(weights_path)    
    print(model.load_state_dict(state_dict, strict=False))
    return model
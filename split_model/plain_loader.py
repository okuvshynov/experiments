import sys
import torch
import json
import os

# use ones from llama2.c
sys.path.insert(0, '../llama2.c')
from model import Transformer, ModelArgs

llama2_7b_path = sys.argv[1]
params_path = os.path.join(llama2_7b_path, "params.json")
weights_path = os.path.join(llama2_7b_path, "consolidated.00.pth")

vocab_size = 32000

def prepare_llama_config():
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
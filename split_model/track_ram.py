import torch
import sys
import json
import time
import gc

#from phantom_model import Transformer, ModelArgs
# use ones from llama2.c
sys.path.insert(0, '../llama2.c')
from model import Transformer, ModelArgs

from phantom_model import Transformer as PhantomTransformer, ModelArgs as PhantomModelArgs

def load_llama7b():
    vocab_size = 32000
    with open(sys.argv[2], 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    model_conf = ModelArgs(**config)
    model = Transformer(model_conf)
    return model

def load_phantom_llama7b():
    vocab_size = 32000
    with open(sys.argv[2], 'r') as conf_file:
        config = json.loads(conf_file.read())
    
    config['vocab_size'] = vocab_size
    model_conf = PhantomModelArgs(**config)
    model = PhantomTransformer(model_conf)
    return model

print('loading llama7b model')
start = time.time()
model = load_llama7b()
print(f'model loaded in {time.time() - start} seconds')
time.sleep(10)
print('deleting model')
del model
print('model deleted')

start = time.time()
print('loading phantom llama7b model')
model = load_phantom_llama7b()
print(f'model loaded in {time.time() - start} seconds')
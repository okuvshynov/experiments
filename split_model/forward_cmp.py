import torch
import time

from phantom_loader import llama7b_phantom
from plain_loader import llama7b_torch

device = 'cpu'
batch_size = 1
X = torch.arange(500 * batch_size).view(batch_size, 500).to(device)

## Running phantom
start = time.time()
model = llama7b_phantom().to(device)
print(f'loaded phantom model in {time.time() - start} seconds')
start = time.time()
phantom_y = model(X)
print(f'evaluated phantom model on batch_size={batch_size} in {time.time() - start} seconds')

## Running default
X = X.to('cpu')
start = time.time()
model = llama7b_torch().to('cpu')
print(f'loaded model in {time.time() - start} seconds')
start = time.time()
plain_y = model(X)
print(f'evaluated model in {time.time() - start} seconds')

same_y = torch.allclose(phantom_y.cpu(), plain_y.cpu())

print(f'output for default torch model and phantom model is same: {same_y}')


"""
Output from Apple M2 @ 24Gb RAM running on MPS and CPU:

% python split_model/manual_load.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
loaded phantom model in 88.39778780937195 seconds
tensor([[[-7.7749,  0.2068,  3.7988,  ..., -2.6116, -3.5703, -1.7604]]],
       device='mps:0', grad_fn=<LinearBackward0>)
evaluated phantom model in 13.354475975036621 seconds
loaded model in 93.51274299621582 seconds
tensor([[[-7.7749,  0.2068,  3.7988,  ..., -2.6116, -3.5703, -1.7604]]],
       grad_fn=<UnsafeViewBackward0>)
evaluated model in 129.28788685798645 seconds
output for default torch model and phantom model is same: False


Looks like the output is not exactly the same but very close - 
I wonder if it's device/precision thing. 
Let's check with running both on CPU  

When both running on CPU we got match.

...
evaluated model in 135.04086709022522 seconds
output for default torch model and phantom model is same: True
"""

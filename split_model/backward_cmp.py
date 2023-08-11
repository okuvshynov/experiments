import time
import torch

from phantom_loader import llama7b_phantom
from plain_loader import llama7b_torch

batch_size = 32
length = 50

def phantom_backwards(device='cpu'):
    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1
    start = time.time()
    model = llama7b_phantom().to(device)
    print(f'loaded phantom model in {time.time() - start} seconds')

    start = time.time()
    # insane LR to see difference after 1 iteration with 1 sample
    opt = torch.optim.SGD(model.parameters(), lr=100.0)

    start = time.time()
    _logits = model(X, Y)
    print(f'phantom forward pass in {time.time() - start} seconds')
    layer_13 = model.layers[13].loaded_inner()
    #print(layer_13.attention.wq.weight)
    weight_before = layer_13.attention.wq.weight.clone()

    start = time.time()
    opt.zero_grad()
    loss = model.last_loss
    loss.backward()
    opt.step()
    print(f'phantom backward pass in {time.time() - start} seconds')

    layer_13 = model.layers[13].loaded_inner()
    weight_after = layer_13.attention.wq.weight.clone()
    return weight_before, weight_after

def plain_backwards(device='cpu'):
    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    start = time.time()
    model = llama7b_torch().to(device)
    print(f'loaded plain model in {time.time() - start} seconds')

    start = time.time()
    # insane LR to see difference after 1 iteration with 1 sample
    opt = torch.optim.SGD(model.parameters(), lr=100.0)

    start = time.time()
    _logits = model(X, Y)
    print(f'plain forward pass in {time.time() - start} seconds')
    #print(layer_13.attention.wq.weight)
    weight_before = model.layers[13].attention.wq.weight.clone()

    start = time.time()
    opt.zero_grad()
    loss = model.last_loss
    loss.backward()
    opt.step()
    print(f'plain backward pass in {time.time() - start} seconds')

    weight_after = model.layers[13].attention.wq.weight.clone()
    return weight_before, weight_after

wb_phantom, wa_phantom = phantom_backwards('mps')
wb_phantom, wa_phantom = phantom_backwards('cpu')
wb_plain, wa_plain = plain_backwards()

same_before = torch.allclose(wb_phantom.cpu(), wb_plain.cpu())
same_after = torch.allclose(wa_phantom.cpu(), wa_plain.cpu())

print(f'{same_after} and {same_before}')

"""
cpu:
 
for batch_size = 1 on m2@24Gb RAM
loaded phantom model in 89.78139281272888 seconds
phantom forward pass in 10.826918840408325 seconds
phantom backward pass in 63.627501010894775 seconds
loaded plain model in 101.10796880722046 seconds
plain forward pass in 127.43440103530884 seconds
plain backward pass in 156.21902322769165 seconds
True and True

for batch_size = 1 on apple m1@16Gb RAM
loaded phantom model in 103.29575395584106 seconds
phantom forward pass in 10.221296072006226 seconds
phantom backward pass in 65.4597909450531 seconds
loaded plain model in 309.6808431148529 seconds
plain forward pass in 120.3576111793518 seconds
zsh: killed     python split_model/backward_cmp.py ../llama/llama-2-7b

"""
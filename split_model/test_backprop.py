# this is an integration test which compares phantom mode vs regular torch execution
# we run forward/backward pass and compare outputs and weights after one optimizer step.

import time
import torch
import sys

from phantom_loader import llama7b_phantom
from plain_loader import llama7b_torch

batch_size = 1
length = 50
seed = 123001
dropout = 0.1

model_path = sys.argv[1]
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'

def phantom_backwards(device='cpu'):
    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    start = time.time()
    model = llama7b_phantom(model_path, dropout=dropout).to(device)
    print(f'loaded phantom model in {time.time() - start} seconds')

    start = time.time()
    # insane LR to see difference after 1 iteration with 1 sample
    opt = torch.optim.SGD(model.parameters(), lr=100.0)

    start = time.time()

    torch.random.manual_seed(seed)
    logits = model(X, Y)
    print(f'phantom forward pass in {time.time() - start} seconds')
    layer_13 = model.layers[13].loaded_inner()
    weight_before = layer_13.attention.wq.weight.clone()

    start = time.time()
    opt.zero_grad()
    loss = model.last_loss
    loss.backward()
    opt.step()
    print(f'phantom backward pass in {time.time() - start} seconds')

    layer_13 = model.layers[13].loaded_inner()
    weight_after = layer_13.attention.wq.weight.clone()
    print(weight_after)
    return weight_before, weight_after, logits.clone()

def plain_backwards(device='cpu'):
    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    start = time.time()
    model = llama7b_torch(model_path, dropout=dropout).to(device)
    print(f'loaded plain model in {time.time() - start} seconds')

    start = time.time()
    # insane LR to see difference after 1 iteration with 1 sample
    opt = torch.optim.SGD(model.parameters(), lr=100.0)

    start = time.time()
    torch.random.manual_seed(seed)
    logits = model(X, Y)
    print(f'plain forward pass in {time.time() - start} seconds')
    weight_before = model.layers[13].attention.wq.weight.clone()

    start = time.time()
    opt.zero_grad()
    loss = model.last_loss
    loss.backward()
    opt.step()
    print(f'plain backward pass in {time.time() - start} seconds')

    weight_after = model.layers[13].attention.wq.weight.clone()
    print(weight_after)
    return weight_before, weight_after, logits.clone()

if __name__ == '__main__':
    print('Running phantom on CPU')
    wb_phantom, wa_phantom, y_phantom = phantom_backwards('cpu')

    print('Running plain on CPU')
    wb_plain, wa_plain, y_plain = plain_backwards()

    same_before = torch.allclose(wb_phantom.cpu(), wb_plain.cpu())
    same_after = torch.allclose(wa_phantom.cpu(), wa_plain.cpu())
    same_y = torch.allclose(y_phantom.cpu(), y_plain.cpu())
    txt = lambda ok: '[ OK ]' if ok else '[FAIL]'

    print(f'{txt(same_before)} weights before')
    print(f'{txt(same_after)} weights after')
    print(f'{txt(same_y)} out logits')

    if device != 'cpu':
        print(f'Running phantom on {device}')
        _, _, _ = phantom_backwards(device)


"""
python split_model/backward_cmp.py ../llama-2-7b
Running phantom on MPS
Loaded 292 module metadata
Created blank model
processing transformer blocks ................................ DONE
populated all weights to model
loaded phantom model in 89.29663586616516 seconds
phantom forward pass in 20.68228006362915 seconds
phantom backward pass in 98.99742364883423 seconds
Running phantom on CPU
Loaded 292 module metadata
Created blank model
processing transformer blocks ................................ DONE
populated all weights to model
loaded phantom model in 89.64134502410889 seconds
phantom forward pass in 33.529903173446655 seconds
phantom backward pass in 129.32752394676208 seconds
Running plain on CPU
loaded plain model in 96.33211588859558 seconds
torch.Size([32, 50, 32000]) torch.Size([1600, 32000]) torch.Size([1600])
plain forward pass in 151.54023575782776 seconds
plain backward pass in 210.3796899318695 seconds
[ OK ] weights before
[ OK ] weights after
[ OK ] out logits

"""

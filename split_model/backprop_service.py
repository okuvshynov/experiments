import time

start = time.time()
import torch
import sys

from utils import intermediate_path
print(f'imports done in {time.time() - start} seconds')

device, module_id, input_id, grad_output_id, grad_input_id, freqs_cos_id, freqs_sin_id = sys.argv[1:]

lr = 100.0

start = time.time()
module = torch.load(intermediate_path(module_id), map_location=torch.device(device))
print(f'module load done in {time.time() - start} seconds')

start = time.time()
input = torch.load(intermediate_path(input_id), map_location=torch.device(device))
print(f'input load done in {time.time() - start} seconds')

# TODO: this can be precomputed, no need to pass around
start = time.time()
freqs_cos = torch.load(intermediate_path(freqs_cos_id), map_location=torch.device(device))
freqs_sin = torch.load(intermediate_path(freqs_sin_id), map_location=torch.device(device))
print(f'constants load done in {time.time() - start} seconds')
input.requires_grad = True

start = time.time()
opt = torch.optim.SGD(module.parameters(), lr=lr)
opt.zero_grad()
print(f'opt/zero grad done in {time.time() - start} seconds')

start = time.time()
grad_output = torch.load(intermediate_path(grad_output_id), map_location=torch.device(device))
print(f'grad_output load done in {time.time() - start} seconds')

start = time.time()
output = module(input, freqs_cos, freqs_sin)
print(f'forward pass done in {time.time() - start} seconds')

start = time.time()
output.backward(grad_output)
print(f'backward pass done in {time.time() - start} seconds')

start = time.time()
opt.step()
print(f'opt.step() done in {time.time() - start} seconds')

start = time.time()
torch.save(input.grad, intermediate_path(grad_input_id))
print(f'saving grad done in {time.time() - start} seconds')

start = time.time()
torch.save(module, intermediate_path(module_id))
print(f'saving module done in {time.time() - start} seconds')
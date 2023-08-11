import torch
import sys

from utils import intermediate_path

module_id, input_id, grad_output_id, grad_input_id, freqs_cos_id, freqs_sin_id = sys.argv[1:]

lr = 100.0

module = torch.load(intermediate_path(module_id))
input = torch.load(intermediate_path(input_id))
freqs_cos = torch.load(intermediate_path(freqs_cos_id))
freqs_sin = torch.load(intermediate_path(freqs_sin_id))
input.requires_grad = True

opt = torch.optim.SGD(module.parameters(), lr=lr)
opt.zero_grad()

grad_output = torch.load(intermediate_path(grad_output_id))
output = module(input, freqs_cos, freqs_sin)
output.backward(grad_output)

opt.step()

torch.save(input.grad, intermediate_path(grad_input_id))
torch.save(module, intermediate_path(module_id))

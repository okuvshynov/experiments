import torch
import sys
import os

module_id, input_id, grad_output_id, grad_input_id = sys.argv[1:]

lr = 0.0001

def print_grad_fn(tensor, indent=""):
    if tensor is None:
        return
    print(indent, tensor.grad_fn)
    for next_function, _ in tensor.grad_fn.next_functions:
        print_grad_fn(next_function, indent + "    ")

def intermediate_path(id):
    if torch.is_tensor(id):
        id = id.item()
    return f'{os.path.dirname(__file__)}/data/saved_{id}.pt'

def id_tensor(id):
    return torch.tensor(id, dtype=torch.int64)

module = torch.load(intermediate_path(module_id))
input = torch.load(intermediate_path(input_id))
input.requires_grad = True

opt = torch.optim.SGD(module.parameters(), lr=lr)
opt.zero_grad()

grad_output = torch.load(intermediate_path(grad_output_id))
output = module(input)
output.backward(grad_output)
opt.step()

torch.save(input.grad, intermediate_path(grad_input_id))
torch.save(module, intermediate_path(module_id))
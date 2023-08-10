import torch
from torch.autograd import Function
from torch.nn import Module
import os

import subprocess
from utils import intermediate_path, random_id

def backwards_call(params):
    path = f'{os.path.dirname(__file__)}/backwards_manually.py'
    subprocess.call(['python', path] + params)

class PrefetchedFn(Function):
    @staticmethod
    def forward(ctx, module_id, input):
        input_id = random_id()
        torch.save(input, intermediate_path(input_id))
        ctx.save_for_backward(module_id, input_id)
        module = torch.load(intermediate_path(module_id))
        output = module(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        module_id, input_id = ctx.saved_tensors
        grad_output_id = random_id()
        grad_input_id = random_id()
        params = [f'{t.item()}' for t in [module_id, input_id, grad_output_id, grad_input_id]]
        torch.save(grad_output, intermediate_path(grad_output_id))
        backwards_call(params)
        return None, torch.load(intermediate_path(grad_input_id))


class PrefetchedModule(Module):
    def __init__(self, module):
        super().__init__()
        self.module_id = random_id()
        torch.save(module, intermediate_path(self.module_id))
        # somewhere here we unload module? and call gc.collect?

    def loaded_inner(self):
        return torch.load(intermediate_path(self.module_id))

    ## maybe no need for forward hook at all? 
    def forward(self, input):
        return PrefetchedFn.apply(self.module_id, input)
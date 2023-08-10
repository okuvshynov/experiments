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
    def forward(ctx, module_id, input, freqs_cos, freqs_sin):
        input_id = random_id()
        torch.save(input, intermediate_path(input_id))
        ctx.save_for_backward(module_id, input_id, freqs_cos, freqs_sin)
        module = torch.load(intermediate_path(module_id), map_location=torch.device('cpu'))
        output = module(input, freqs_cos, freqs_sin)
        return output

    # TODO backwards call is wrong here
    @staticmethod
    def backward(ctx, grad_output):
        module_id, input_id, freqs_cos, freqs_sin = ctx.saved_tensors
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

    def loaded_inner(self):
        return torch.load(intermediate_path(self.module_id))
    
    def populate_state_dict(self, state_dict):
        module = self.loaded_inner()

        # fix shapes before loading
        def fix_shapes_rec(module, prefix=''):
            nonlocal state_dict
            if hasattr(module, 'weight'):
                key = f'{prefix}weight'
                loaded_shape = state_dict[key].shape
                state_dict[key] = state_dict[key].reshape(module.weight.shape)

            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    fix_shapes_rec(child, child_prefix)

        fix_shapes_rec(module)
        module.load_state_dict(state_dict)
        torch.save(module, intermediate_path(self.module_id))

    ## maybe no need for forward hook at all? 
    def forward(self, input, freqs_cos, freqs_sin):
        return PrefetchedFn.apply(self.module_id, input, freqs_cos, freqs_sin)
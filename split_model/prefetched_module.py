import torch
import os

import subprocess
from utils import intermediate_path, random_id

def backwards_call(device, params):
    path = f'{os.path.dirname(__file__)}/backprop_service.py'
    subprocess.call(['python', path, device] + params)

def device_map(device):
    if str(device).startswith('mps'):
        return 'mps'
    return str(device)

class PrefetchedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module_id, input, freqs_cos, freqs_sin):
        device = device_map(input.device)

        input_id = random_id()
        torch.save(input, intermediate_path(input_id))
        ctx.save_for_backward(module_id, input_id, freqs_cos, freqs_sin)
        module = torch.load(intermediate_path(module_id), map_location=torch.device(device))
        output = module(input, freqs_cos, freqs_sin)
        return output

    # as a first step just pass cos/sin as well. later we should just load them to backward service
    @staticmethod
    def backward(ctx, grad_output):
        module_id, input_id, freqs_cos, freqs_sin = ctx.saved_tensors
        device = device_map(grad_output.device)
        grad_output_id = random_id()
        grad_input_id = random_id()
        freqs_sin_id = random_id()
        freqs_cos_id = random_id()
        params = [f'{t.item()}' for t in [module_id, input_id, grad_output_id, grad_input_id, freqs_cos_id, freqs_sin_id]]
        torch.save(grad_output, intermediate_path(grad_output_id))
        torch.save(freqs_cos, intermediate_path(freqs_cos_id))
        torch.save(freqs_sin, intermediate_path(freqs_sin_id))
        
        backwards_call(device, params)
        return None, torch.load(intermediate_path(grad_input_id), map_location=torch.device(device)), None, None

class PrefetchedModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module_id = random_id()
        torch.save(module, intermediate_path(self.module_id))

    def loaded_inner(self):
        return torch.load(intermediate_path(self.module_id))
    
    def from_state_dict(self, state_dict):
        module = self.loaded_inner()

        # fix shapes before loading
        def fix_shapes_rec(module, prefix=''):
            nonlocal state_dict
            if hasattr(module, 'weight'):
                key = f'{prefix}weight'
                state_dict[key] = state_dict[key].reshape(module.weight.shape)

            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    fix_shapes_rec(child, child_prefix)

        fix_shapes_rec(module)
        module.load_state_dict(state_dict)
        torch.save(module, intermediate_path(self.module_id))

    def to_state_dict(self):
        return self.loaded_inner().state_dict()

    def forward(self, input, freqs_cos, freqs_sin):
        return PrefetchedFn.apply(self.module_id, input, freqs_cos, freqs_sin)
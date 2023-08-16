import torch

from utils import intermediate_path, save_rng_state
from backprop_service import Backprop

def device_map(device):
    if str(device).startswith('mps'):
        return 'mps'
    return str(device)

global_id_auto = 0

def next_id():
    global global_id_auto
    res = torch.tensor(global_id_auto)
    global_id_auto += 1
    return res


def backwards_call(device, params):
    if not hasattr(backwards_call, 'backprop'):
        backwards_call.backprop = Backprop()
        
    params = [device] + params
    backwards_call.backprop.run(params)

class BlackboxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module_id, input, freqs_cos, freqs_sin, is_training, input_id, grad_input_id, grad_output_id):
        device = device_map(input.device)

        torch.save(input, intermediate_path(input_id))

        # we need to save rng state here as well to do second forward pass exactly the same
        ctx.save_for_backward(module_id, input_id, freqs_cos, freqs_sin, save_rng_state(device), grad_input_id, grad_output_id)
        module = torch.load(intermediate_path(module_id), map_location=torch.device(device))
        if not is_training:
            module.eval()
        output = module(input, freqs_cos, freqs_sin)
        return output

    # as a first step just pass cos/sin as well. later we should just load them to backward service
    @staticmethod
    def backward(ctx, grad_output):
        module_id, input_id, freqs_cos, freqs_sin, rng_state, grad_input_id, grad_output_id = ctx.saved_tensors
        device = device_map(grad_output.device)
        params = [module_id.item(), input_id.item(), grad_output_id.item(), grad_input_id.item(), freqs_cos.to('cpu'), freqs_sin.to('cpu'), rng_state.to('cpu')]
        torch.save(grad_output, intermediate_path(grad_output_id))
        
        backwards_call(device, params)
        return None, torch.load(intermediate_path(grad_input_id), map_location=torch.device(device)), None, None, None, None, None, None

class Blackbox(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module_id = next_id()
        self.input_id = next_id()
        self.grad_input_id = next_id()
        self.grad_output_id = next_id()
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
        return BlackboxFn.apply(self.module_id, input, freqs_cos, freqs_sin, self.training, self.input_id, self.grad_input_id, self.grad_output_id)

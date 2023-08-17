import torch

from utils import intermediate_path, save_rng_state, device_map, next_id
from backprop_service import Backprop

def backwards_call(device, params):
    if not hasattr(backwards_call, 'backprop'):
        backwards_call.backprop = Backprop()
        
    params = [device] + params
    return backwards_call.backprop.run(params)

class BlackboxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module_id, is_training, input_id, input, *args):
        device = device_map(input.device)

        torch.save(input, intermediate_path(input_id))

        # we need to save rng state here as well to do second forward pass exactly the same
        ctx.save_for_backward(module_id, input_id, save_rng_state(device), *args)
        module = torch.load(intermediate_path(module_id), map_location=torch.device(device))
        if not is_training:
            module.eval()
        output = module(input, *args)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        module_id, input_id, rng_state, *extra = ctx.saved_tensors
        device = device_map(grad_output.device)

        params = [module_id.item(), input_id.item(), grad_output.to('cpu'), rng_state.to('cpu')]
        extra = [t.to('cpu') for t in extra]
        
        grad_input = backwards_call(device, params + extra).to(device)
        return None, None, None, grad_input, None, None

class Blackbox(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module_id = next_id()
        self.input_id = next_id()
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

    def forward(self, input, *args):
        return BlackboxFn.apply(self.module_id, self.training, self.input_id, input, *args)

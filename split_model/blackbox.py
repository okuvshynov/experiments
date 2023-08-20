import torch

from utils import intermediate_path, device_map, next_id

class Blackbox(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module_id = next_id()
        self.input_id = next_id()
        torch.save(module, intermediate_path(self.module_id))

    def loaded_inner(self):
        return torch.load(intermediate_path(self.module_id))
    
    def load(self, device):
        return torch.load(intermediate_path(self.module_id), map_location=torch.device(device_map(device)))

    def save(self, module):
        torch.save(module, intermediate_path(self.module_id))
    
    def load_input(self, device):
        return torch.load(intermediate_path(self.input_id), map_location=torch.device(device_map(device)))
    
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
        torch.save(input, intermediate_path(self.input_id))
        device = device_map(input.device)
        module = torch.load(intermediate_path(self.module_id), map_location=torch.device(device))

        if not self.training:
            module.eval()
        
        # we offload model anyway. for backwards pass we do it manually.
        # no need to have gradient here ever.
        with torch.no_grad():
            return module(input, *args)

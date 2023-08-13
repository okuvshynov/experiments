import torch
import os

def intermediate_path(id):
    if torch.is_tensor(id):
        id = id.item()
    return f'{os.path.dirname(__file__)}/data/saved_{id}.pt'

# TODO: cuda is probbaly wrong here, need to test
def save_rng_state(device='cpu'):
    if device == 'cpu':
        import torch
        return torch.random.get_rng_state()
    elif device.startswith('cuda'):
        return torch.cuda.get_rng_state(device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        return torch.mps.get_rng_state()
    else:
        raise ValueError(f"Unsupported device: {device}")

def restore_rng_state(rng_state, device='cpu'):
    if device == 'cpu':
        import torch
        torch.random.set_rng_state(rng_state)
    elif device.startswith('cuda'):
        torch.cuda.set_rng_state(rng_state, device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        torch.mps.set_rng_state(rng_state)
    else:
        raise ValueError(f"Unsupported device: {device}")
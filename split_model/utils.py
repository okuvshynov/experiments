import torch
import os

def random_id():
    return torch.randint(torch.iinfo(torch.int64).max, (1, ), dtype=torch.int64)

def intermediate_path(id):
    if torch.is_tensor(id):
        id = id.item()
    return f'{os.path.dirname(__file__)}/data/saved_{id}.pt'
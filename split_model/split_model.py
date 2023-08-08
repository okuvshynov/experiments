import torch
import torch.nn as nn
from torch.nn import functional as F
from prefetched_module import PrefetchedModule

#torch.manual_seed(18887)

width = 2
n_layers = 2
lr = 0.0001
batch_size = 1
X = torch.rand(batch_size, width)
X.requires_grad = True
target = torch.rand(batch_size, width)

# assume we loaded weights for 2 layers
weights = [torch.rand(width, width), torch.rand(width, width)]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width, bias=False),
            nn.Tanh(),
            nn.Linear(width, width, bias=False),
            nn.Tanh(),
        )
        self.net[0].weight.data = weights[0].clone()
        self.net[2].weight.data = weights[1].clone()
        
    def forward(self, x):
        return self.net(x)
    
class PrefetchedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width, bias=False),
            nn.Tanh(),
            nn.Linear(width, width, bias=False),
            nn.Tanh(),
        )
        self.net[0].weight.data = weights[0].clone()
        self.net[2].weight.data = weights[1].clone()

        self.net[0] = PrefetchedModule(self.net[0])
        self.net[2] = PrefetchedModule(self.net[2])

        
    def forward(self, x):
        return self.net(x)


# regular model
model = Model()
y = model(X)
opt = torch.optim.SGD(model.parameters(), lr=lr)
opt.zero_grad()
loss1 = F.mse_loss(target, y)
loss1.backward()
opt.step()

# cut model
model_unload = PrefetchedModel()
y = model_unload(X)
loss1 = F.mse_loss(target, y)
loss1.backward()

print('Comparing weights to unloadable wrapper')
print(torch.all(torch.isclose(model_unload.net[0].loaded_inner().weight.data, model.net[0].weight.data)).item())
print(torch.all(torch.isclose(model_unload.net[2].loaded_inner().weight.data, model.net[2].weight.data)).item())

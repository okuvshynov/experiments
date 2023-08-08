# just a small test to make sure I understand how it works.
# we create 2 models:
# 1 with 2 layers

# another one split into 2 chunks but initialized the same
# after that, we run a step of training on the same data with the same parameters and
# make sure the weights are updated the same way

import torch
import torch.nn as nn
from torch.nn import functional as F
import io
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

class Layer(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.net = nn.Linear(width, width, bias=False)
        self.tanh = nn.Tanh()
        self.net.weight.data = weights.clone()

    def forward(self, x):
        return self.tanh(self.net(x))

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

class Chunk:
    def __init__(self, module):
        self.xbuf = io.BytesIO()
        self.buf = io.BytesIO()
        torch.save(module, self.buf)

    def forward(self, X):
        torch.save(X, self.xbuf)

        module = self.load_module()
        res = module(X).detach()
        res.requires_grad = True
        return res
    
    def load_module(self):
        self.buf.seek(0)
        return torch.load(self.buf)
    
    def load_input(self):
        self.xbuf.seek(0)
        return torch.load(self.xbuf)

    # returns grad 
    def backward(self, grad):
        # run forward again
        x = self.load_input()
        x.requires_grad = True

        module = self.load_module()
        out = module(x)

        # stateless optimizer
        opt = torch.optim.SGD(module.parameters(), lr=lr)
        opt.zero_grad()

        # now backwards
        out.backward(grad)

        # save gradients instead? need for gradient accumulation
        opt.step()

        # saving with updated weights
        self.buf.seek(0)
        torch.save(module, self.buf)

        return x.grad

# combined model
model = Model()
y = model(X)
opt = torch.optim.SGD(model.parameters(), lr=lr)
opt.zero_grad()
loss1 = F.mse_loss(target, y)
loss1.backward()
opt.step()

chunks = [Chunk(Layer(w)) for w in weights]
curr = X

# forward pass. We don't really need grad here
# but we don't do eval() either, as layers like dropout
# operate differently at training/eval modes
for chunk in chunks:
    curr = chunk.forward(curr)

# compute the loss function. Curr here will require gradient computed
loss = F.mse_loss(target, curr)
loss.backward()

# compute gradients and adjust weights in reverse order
curr_grad = curr.grad
for chunk in reversed(chunks):
    curr_grad = chunk.backward(curr_grad)


# check that weights are same after one optimizer step
# in combined model and by-chunk
print('Comparing to manually chunked')
print(torch.all(torch.isclose(model.net[0].weight.data, chunks[0].load_module().net.weight.data)).item())
print(torch.all(torch.isclose(model.net[2].weight.data, chunks[1].load_module().net.weight.data)).item())

model_unload = PrefetchedModel()
y = model_unload(X)
loss1 = F.mse_loss(target, y)
loss1.backward()

print('Comparing to unloadable wrapper')
print(torch.all(torch.isclose(model_unload.net[0].loaded_inner().weight.data, model.net[0].weight.data)).item())
print(torch.all(torch.isclose(model_unload.net[2].loaded_inner().weight.data, model.net[2].weight.data)).item())

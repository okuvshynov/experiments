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

width = 2
n_layers = 2
lr = 0.0001

torch.manual_seed(18887)

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

batch_size = 1
X = torch.rand(batch_size, width)
target = torch.rand(batch_size, width)

# combined model
model = Model()
y = model(X)
opt = torch.optim.SGD(model.parameters(), lr=lr)
opt.zero_grad()
loss1 = F.mse_loss(target, y)
loss1.backward()
#print(model.net[0].weight.grad)
#print(model.net[2].weight.grad)
opt.step()

#print(model.net[0].weight.data)
#print(model.net[2].weight.data)

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

        # do optimizer step right here? Otherwise gradients are lost. Alternative is to save the module with gradients which seems like an overkill
        #print(module.net.weight.grad)
        opt.step()

        # saving with updated weights
        self.buf.seek(0)
        torch.save(module, self.buf)

        return x.grad

chunks = [Chunk(Layer(w)) for w in weights]

curr = X

for chunk in chunks:
    curr = chunk.forward(curr)

loss = F.mse_loss(target, curr)
loss.backward()

curr_grad = curr.grad
for chunk in reversed(chunks):
    curr_grad = chunk.backward(curr_grad)

# check that weights are same after one optimizer step
print(torch.all(torch.isclose(model.net[0].weight.data, chunks[0].load_module().net.weight.data)).item())
print(torch.all(torch.isclose(model.net[2].weight.data, chunks[1].load_module().net.weight.data)).item())
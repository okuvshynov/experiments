# benchmark we'll use to measure some perf improvements 
# single iteration of forward/backward pass

import time
import torch
import sys

from blackbox_loader import load_llama7b
import backprop_service

batch_size = 2
length = 32
seed = 123001
dropout = 0.1
iters = 2
lr = 1.0
device = 'mps'

if __name__ == '__main__':
    model_path = sys.argv[1]

    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    backprop_service.lr = lr

    start = time.time()
    model = load_llama7b(model_path, dropout=dropout).to(device)
    print(f'loaded model in {time.time() - start} seconds')

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    torch.random.manual_seed(seed)
    for _ in range(iters):
        start = time.time()
        logits = model(X, Y)
        print(f'forward pass in {time.time() - start} seconds')

        start = time.time()
        opt.zero_grad()
        loss = model.last_loss
        loss.backward()
        opt.step()
        print(f'backward pass in {time.time() - start} seconds')

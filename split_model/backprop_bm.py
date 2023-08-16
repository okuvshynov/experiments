# benchmark we'll use to measure some perf improvements 
# single iteration of forward/backward pass

import time
import torch
import sys

from phantom_loader import llama7b_phantom

batch_size = 2
length = 2048
seed = 123001
dropout = 0.1
iters = 2

if __name__ == '__main__':
    model_path = sys.argv[1]
    device = 'mps'

    X = torch.arange(length * batch_size).view(batch_size, length).to(device)
    Y = X + 1

    start = time.time()
    model = llama7b_phantom(model_path, dropout=dropout).to(device)
    print(f'loaded phantom model in {time.time() - start} seconds')

    opt = torch.optim.SGD(model.parameters(), lr=100.0)

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

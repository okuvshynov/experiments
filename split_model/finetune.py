# first iteration of fine-tuning, we don't even save the model anywhere

import os
import requests
import sys
import time
import torch

from phantom_loader import llama7b_phantom

sys.path.insert(0, '../llama/llama')
from tokenizer import Tokenizer

# download tiny dataset
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(url).text

model_path = '../llama-2-7b'
device = 'mps'
split = 0.9
seq_len = 2048
dropout = 0.05
batch_size = 2
seed = 1997
iters = 100

lr = 0.000001

tokenizer_path = os.path.join(model_path, 'tokenizer.model')

tokenizer = Tokenizer(tokenizer_path)
tokens = tokenizer.encode(text, True, True)

n = len(tokens)
idx = int(n * split)

train = tokens[:idx]
val = tokens[idx:]

print(f'loaded datasets: train[{len(train)}], val[{len(val)}]')

# dataset is either train or val
def get_batch(data, batch_size):
    index = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i + seq_len]).to(torch.int64) for i in index])
    y = torch.stack([torch.tensor(data[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
    return x.to(device), y.to(device)

X, y = get_batch(train, batch_size)
print(X, y)

model = llama7b_phantom(model_path, dropout=dropout).to(device)

opt = torch.optim.SGD(model.parameters(), lr=lr)

torch.random.manual_seed(seed)
for i in range(iters):
    start = time.time()
    X, y = get_batch(train, batch_size)
    print(f'got data batch: {time.time() - start}, {X.shape}, {y.shape}')
    logits = model(X, y)
    print(f'forward done: {time.time() - start}')

    opt.zero_grad()
    loss = model.last_loss
    print(f'batch loss: {loss.item()}')
    loss.backward()
    opt.step()
    print(f'backprop done: {time.time() - start}')


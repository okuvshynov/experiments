# first iteration of fine-tuning, we don't even save the model anywhere

import os
import requests
import sys
import time
import torch

from blackbox_loader import load_llama7b

import backprop_service

sys.path.insert(0, '../llama/llama')
from tokenizer import Tokenizer

# download tiny dataset
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(url).text

model_path = '../llama-2-7b'
device = 'mps'
split = 0.9
seq_len = 128
dropout = 0.05
batch_size = 2
seed = 1997
iters = 100
eval_iters = 10
# can afford larger batch size for no-grad 
eval_batch_size = 4
lr = 1e-5

if __name__ == '__main__':
    torch.random.manual_seed(seed)
    backprop_service.lr = lr

    tokenizer_path = os.path.join(model_path, 'tokenizer.model')

    tokenizer = Tokenizer(tokenizer_path)
    tokens = tokenizer.encode(text, True, True)

    n = len(tokens)
    idx = int(n * split)

    train = tokens[:idx]
    val = tokens[idx:]

    print(f'loaded datasets: train[{len(train)}], val[{len(val)}]')

    model = load_llama7b(model_path, dropout=dropout).to(device)

    # dataset is either train or val
    def get_batch(data, batch_size):
        index = torch.randint(len(data) - seq_len, (batch_size,))
        x = torch.stack([torch.tensor(data[i:i + seq_len]).to(torch.int64) for i in index])
        y = torch.stack([torch.tensor(data[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
        return x.to(device), y.to(device)

    def val_loss():
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(eval_iters):
                X, y = get_batch(val, eval_batch_size)
                logits = model(X, y)
                losses.append(model.last_loss)
                print(f'val loss = {model.last_loss}')

        model.train()

    X, y = get_batch(train, batch_size)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    start = time.time()
    for i in range(iters):
        print(f'start iter {i} @ {time.time() - start:.3g}')
        if (i % 2 == 0):
            val_loss()
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

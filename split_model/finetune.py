import os
import requests
import sys
import time
import torch
import logging

from loader import load_llama2, save_llama2

# use tokenizer from llama
sys.path.insert(0, '../llama/llama')
from tokenizer import Tokenizer

# download tiny dataset
url = 'https://www.gutenberg.org/cache/epub/67098/pg67098.txt'
text = requests.get(url).text

model_path = '../llama-2-70b'
new_model_path = '../llama-2-70b-tuned'

split = 0.9
seed = 1997
iters = 1
device = 'mps'

seq_len = 64
dropout = 0.01
batch_size = 8
lr = 1e-5

eval_period = 20
eval_iters = 20

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    torch.random.manual_seed(seed)

    tokenizer_path = os.path.join(model_path, 'tokenizer.model')

    tokenizer = Tokenizer(tokenizer_path)
    tokens = tokenizer.encode(text, True, True)

    n = len(tokens)
    idx = int(n * split)

    train = tokens[:idx]
    val = tokens[idx:]

    logging.info(f'loaded datasets: train[{len(train)}], val[{len(val)}]')

    model = load_llama2(model_path, dropout=dropout).to(device)

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
                X, y = get_batch(val, batch_size)
                logits = model(X, y)
                losses.append(model.last_loss)
                logging.info(f'val loss = {model.last_loss}')

        model.train()

    X, y = get_batch(train, batch_size)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    start = time.time()
    for i in range(iters):
        logging.info(f'{time.time() - start} starting iteration {i}')
        if (i % eval_period == 0 and i > 0):
            val_loss()
        X, y = get_batch(train, batch_size)
        opt.zero_grad()
        # both forward and backward passes are here.
        # returned loss is a scalar, not variable
        logits, loss = model.manual_loop(X, y, lr=lr)
        opt.step()
        logging.info(f'backprop done: {time.time() - start}, loss = {loss}')

    save_llama2(model, new_model_path, model_path, shards=1)
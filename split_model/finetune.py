import os
import sys
import torch
import logging

from loader import load_llama2, save_llama2

# use tokenizer from llama
sys.path.insert(0, '../llama/llama')
from tokenizer import Tokenizer

with open('split_model/test_data/alice.txt') as f:
    text = f.read()

model_path = '../llama-2-13b'
new_model_path = '../llama-2-13b-tuned'
shards_to_save = 2

seed = 1997
iters = 50
device = 'mps'
seq_len = 256
dropout = 0.01
batch_size = 16
lr = 1e-3

eval_period = 10
gen_tokens = 20

tokenizer_path = os.path.join(model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

def greedy_gen(prompt, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(prompt, True, False)).view(1, -1).to(device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        logits_top, next_tokens = torch.topk(logits, k=5, dim=-1)
        next_token = next_tokens[0, 0].view(1, 1)
        logging.info(f'next tokens: {logits_top} {next_tokens} {tokenizer.decode(next_tokens.tolist())}')
        tokens = torch.cat((tokens, next_token), dim=1)

    for i, output in enumerate(tokens):
        logging.info(f'{i} - {tokenizer.decode(output.tolist())}')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    torch.random.manual_seed(seed)

    tokenizer_path = os.path.join(model_path, 'tokenizer.model')

    tokenizer = Tokenizer(tokenizer_path)
    tokens = tokenizer.encode(text, True, True)

    n = len(tokens)
    train = tokens

    logging.info(f'loaded dataset: train[{len(train)}]')

    model = load_llama2(model_path, dropout=dropout).to(device)

    # dataset is either train or val
    def get_batch(batch_size):
        index = torch.randint(len(train) - seq_len, (batch_size,))
        x = torch.stack([torch.tensor(train[i:i + seq_len]).to(torch.int64) for i in index])
        y = torch.stack([torch.tensor(train[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
        return x.to(device), y.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(iters):
        logging.info(f'starting iteration {i}')
        X, y = get_batch(batch_size)
        opt.zero_grad()
        if i % eval_period == 0:
            greedy_gen('Alice drank from the bottle which had a label: ', max_new_tokens=20)
        # both forward and backward passes are here.
        # returned loss is a scalar, not variable
        logits, loss = model.manual_loop(X, y, lr=lr)
        opt.step()
        logging.info(f'backprop done, loss = {loss}')

    save_llama2(model, new_model_path, model_path, shards=shards_to_save)

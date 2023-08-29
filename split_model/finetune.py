import os
import sys
import torch
import logging

from loader import load_llama2, save_llama2

# use tokenizer from llama
sys.path.insert(0, '../llama/llama')
from tokenizer import Tokenizer

# data to finetune on
with open('split_model/test_data/cubestat.txt') as f:
    text = f.read()
prompt = 'Cubestat is a tool which can measure and visualize '

# old/new model paths
model_path = '../llama-2-13b'
new_model_path = '../llama-2-13b-tuned'
shards_to_save = 2

# training settings
seed = 1997
iters = 1000
device = 'mps'
seq_len = 256
dropout = 0.01
batch_size = 16
lr = 3e-3

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
        logits_top, next_tokens = torch.topk(logits, k=25, dim=-1)
        next_token = next_tokens[0, 0].view(1, 1)
        logging.info(f'next tokens: {logits_top} {next_tokens} {tokenizer.decode(next_tokens.tolist())}')
        tokens = torch.cat((tokens, next_token), dim=1)

    for i, output in enumerate(tokens):
        logging.info(f'{i} - {tokenizer.decode(output.tolist())}')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename='finetune.log')
    torch.random.manual_seed(seed)

    tokenizer_path = os.path.join(model_path, 'tokenizer.model')

    tokenizer = Tokenizer(tokenizer_path)
    tokens = tokenizer.encode(text, True, True)

    logging.info(f'loaded dataset: {len(tokens)} tokens')

    model = load_llama2(model_path, dropout=dropout).to(device)

    def get_batch(batch_size):
        index = torch.randint(len(tokens) - seq_len, (batch_size,))
        x = torch.stack([torch.tensor(tokens[i:i + seq_len]).to(torch.int64) for i in index])
        y = torch.stack([torch.tensor(tokens[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
        return x.to(device), y.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(iters):
        logging.info(f'starting iteration {i}')
        X, y = get_batch(batch_size)
        opt.zero_grad()
        if i % eval_period == 0:
            greedy_gen(prompt, max_new_tokens=20)
        # both forward and backward passes are here.
        # returned loss is a scalar, not variable
        logits, loss = model.manual_loop(X, y, lr=lr)
        opt.step()
        logging.info(f'backprop done, loss after forward pass = {loss}')

    greedy_gen(prompt, max_new_tokens=20)
    save_llama2(model, new_model_path, model_path, shards=shards_to_save)

import torch
import sys
import os

sys.path.insert(0, '../llama/llama')
from tokenizer import Tokenizer

from phantom_loader import llama7b_phantom

model_path = sys.argv[1]

tokenizer_path = os.path.join(model_path, 'tokenizer.model')
tokenizer = Tokenizer(tokenizer_path)

model = llama7b_phantom(sys.argv[1], dropout=0.0).to('mps')

def greedy_gen(prompt, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(prompt, True, False)).view(1, -1).to('mps')
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        _, next_token = torch.topk(logits, k=1, dim=-1)
        print(f'next token: {next_token} {tokenizer.decode(next_token.tolist())}')
        tokens = torch.cat((tokens, next_token), dim=1)

    for i, output in enumerate(tokens):
        print(f'{i} - {tokenizer.decode(output.tolist())}')

prompt = 'I believe the meaning of life is'

greedy_gen(prompt, max_new_tokens=10)
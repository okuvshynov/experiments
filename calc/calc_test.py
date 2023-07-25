import pickle
import torch
from contextlib import nullcontext

tokens = [''] + [*'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
shifts = {token: index for index, token in enumerate(tokens) if token != ''}


def cursor_decode(value):
    res = []
    offset = 0
    for chr in value:
        if chr in shifts.keys():
            offset += shifts[chr]
        else:
            res.insert(len(res) - offset, chr)
            offset = 0
    return ''.join(res)

class Evaluator:
    def __init__(self, meta_path=None):
        if meta_path is not None:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.stoi, self.itos = meta['stoi'], meta['itos']
            self.decode = lambda l: ''.join([self.itos[i] for i in l])
            self.encode = lambda s: [self.stoi[c] for c in s]

    def run(self, model, ctx=nullcontext(), iters=1000, digits=4, device='cuda', top_k=20, temp=0.1, max_new_tokens=20, invert=False):
        model.eval()
        mn = 10 ** (digits - 1)
        mx = mn * 10
        with torch.no_grad():
            with ctx:
                correct = 0.0
                for _ in range(iters):
                    A = torch.randint(mn, mx, (1,)).item()
                    B = torch.randint(mn, mx, (1,)).item()
                    prompt = f'{A}+{B}='
                    start_ids = self.encode(prompt)
                    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                    y = model.generate(x, max_new_tokens, temperature=temp, top_k=top_k)
                    try:
                        seq = self.decode(y[0].tolist()).split('\n')[0].split('=')[1]
                        # if there's no cursor, it is no-op
                        seq = cursor_decode(seq)
                        res = int(seq[::-1]) if invert else int(seq)
                        if res == A + B:
                            correct += 1
                    except:
                        # if anything goes wrong, it was not correct
                        pass
        model.train()
        return correct / iters

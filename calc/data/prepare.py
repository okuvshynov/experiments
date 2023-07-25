import os
import pickle
import numpy as np
import torch
import sys

def cursor_encode(value):
    res = ''

    tokens = [''] + [*'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    for (c, d) in zip(value[::-1], tokens):
        res += d + c
    return res

def add_cursor(value):
    delim = '<'
    curr = ''
    for ch in value[::-1]:
        curr += ch + delim
        delim += '<'
    
    return curr[:len(curr) - len(delim) + 1]


def prepare(mode='', digits=4, samples=10000):
    if mode not in ['', 'inverted', 'cursor']:
        print('mode mush be "inverted" or "cursor"')
        exit(1)

    
    data = ''
    masks = []
    mn = 10 ** (digits - 1)
    mx = mn * 10

    for _ in range(samples):
        A = torch.randint(mn, mx, (1,)).item()
        B = torch.randint(mn, mx, (1,)).item()
        ab = f'{A}+{B}='

        c = f'{A+B}'
        if mode == 'cursor':
            c = cursor_encode(c)
        if mode == 'inverted':
            c = c[::-1]
        
        res = f'{ab}{c}\n'

        mask = [0] * len(ab) + [1] * len(c) + [1] # learning the result and end of line.
        data = data + res
        masks = masks + mask


    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s]

    # create the train and test splits
    n = len(data)

    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    train_masks = masks[:int(n * 0.9)]
    val_masks = masks[int(n * 0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f'train_mask is {len(train_masks)}')
    print(f'val_mask is {len(val_masks)}')

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_masks = np.array(train_masks, dtype=np.uint16)
    val_masks = np.array(val_masks, dtype=np.uint16)

    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    train_masks.tofile(os.path.join(os.path.dirname(__file__), 'train_masks.bin'))
    val_masks.tofile(os.path.join(os.path.dirname(__file__), 'val_masks.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

mode = sys.argv[1] if len(sys.argv) > 1 else ''
prepare(mode=mode, digits=5)
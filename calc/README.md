### Training tiny language model to learn addition 

Trains tiny transformer model with 3 vairants of data format:
1. Regular: 345+678=1023
2. Inverted results: 345+678=3201
3. Inverted + cursor: 345+678=3<2<<0<<<1

Also optionally masking loss for output only, so that we propagate the loss only for 'results' rather than operands. 

Based on https://github.com/karpathy/nanoGPT

Licence: MIT

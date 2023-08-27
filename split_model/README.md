## slowllama

Fine-tune Llama2 models, including 70B on Apple M1/M2 devices.

slowllama is not using any quantization, no qlora, etc. Instead, it offloads parts of model to/from SSD on both forward/backward passes. In contrast with training large models from scratch (unattainable) or inference (where we are likely to care about interactivity and tokens/sec), we can still get something finetuned if we allow it to run, say, overnight in batches of modest size. The model is saved back to the same format with same types, layers and number of shards as original llama2.

NOTE: SSDs can handle limited number of writes during lifecycle. I don't know if this number for MacBooks is published anywhere, but this is something to be aware of, use at your own risk. Once LoRA is implemented, this should be mush less of an issue.

slowllama is most definitely not suitable for anything research-like with heavy experimentation as it is too slow - the duration of iteration cycle would kill the productivity. The use-case here is rather to be part of a product which makes small changes based on personal/local data, for example set of documents or code someone is working on.

Finetuning is the only focus here, there's nothing special done for inference, refer to [llama.cpp](https://github.com/ggerganov/llama.cpp) for that.

### Example

Let's start with an example: [a subset of public-domain book](test_data/alice.txt). It is probably small enough to just be included as part of the prompt, but it's a decent illustration. Asking llama2-13b to complete the prompt "Alice drank from the bottle which had a label " gives a continuation "100% Pure Maple Syrup.". 

In order to fine-tune llama2 model we need to:
1. Install dependencies: ```pip install torch sentencepiece``` 
2. Clone [llama2](https://github.com/facebookresearch/llama) and follow its instructions to download the models. It will download tokenizer as well. tokenizer.model should be put into the same directory as llama model itself. Example folder structure could look like:
```
/parent/
    /llama-2-7b/... 
    /llama-2-13b/...
    /llama-2-70b/...
    /llama/...     # <-- this is Meta's llama2 repository
    /slowllama/... # <- this repo
```
If we finetune the llama2-13b model for ~1 hour on MacMini M1 with 16Gb RAM:
```
pip install torch 
pip install sentencepiece
python finetune.py
```

we see improvement in loss 

<TBD>

and much better continuation is produced: 

"Alice drank from the bottle which had a label: Drink me, and she grew "

[finetune.py](finetune.py) is a very simple script which finetunes based on the plaintext data. There are some settings you could change here, like sequence length, batch size, learning rate, dropout rate, number of iterations. Change this if desired. Model path for input/output is hardcoded in that script as well, change accordingly. For the model saving, double-check that you use the expected number of shards (TODO: just reuse one from original model). 

### How does it work?
Most of the tests were done with llama2-13b. 7B model will have one shard and 70B model will have 8, but other than that there's nothing too special.

First, we need to be able to load a model which takes more RAM than we have. We create model instance with all large modules' weights offloaded to SSD - for each of the transformer blocks, token embeddings and output linear layer. After that we load model shards one by one, for each shard iterate over all modules, update corresponding subset of its weights and save it back. 

Original llama2 weights are in bfloat16, but mps backend doesn't support that type natively, so we do computation in float32 instead.

Doing forward path is easy - we just load each module when we need, evaluate it and propagate the result. Backward pass is a little more tricky. The way it's currently implemented is:
1. Do a forward pass the same way as above, while also saving inputs to each block to the hard drive.
2. Then, do a manual backward gradient propagation. We start from the end, and for each offloaded block re-run it with the same input again. We run backward pass within that block, update the weights for that block, save it back to hard drive and pass the gradient for the input to the next (previous?) module. Repeat. Important: we also need to save and restore random number generation state. During training we use dropout, and randomly switched off neurons should be the same on both forward passes.

### Resource utilization/requirements/limitations

1. Only stateless optimizer for now (SGD). We'll have to load/save AdamW state repeatedly as well to be able to use it;
2. It is slow, but GPU is reasonably well utilized;
3. SSD requirements - even for 13B model you'll need: 26-27Gb for original weights + 50Gb for intermediate weights (in float32) + same 26-27Gb if you want to save a single snapshot, so ~100-110Gb total. You are looking at something order of 500Gb for llama70. TODO: measure how many writes do we do exactly and if we significantly impact SSD lifetime here? Once LoRA is implemented this should be less of an issue;
4. no gradient accumulation, we just update weights as we go. Accumulating gradient would require saving gradients as well.

insert charts from cubestat here

Even currently available maxed-out m2 macbook pro should have ~5x GPU performance, so loading/saving might become much more noticable. To improve on that we can:
1. Use LoRA and only update a few tranable parameters.
2. prefetch/save layer asynchronously.


### Files

```
blackbox_model.py -- model definition and manual backprop implementation
finetune.py - script which does the training
loader.py - manual loading/saving of large llama2 models (including 70B)
utils.py - small utility functions, including saving/loading random generator state for different devices.

test_backprop.py - test which loads a 7b model, runs one iteration of backprop on hardcoded data and compares the outputs/weights.
test_gen.py - complete the sequence. Useful for sanity checks.
test_ref_loader.py - loader for reference implementation from llama2.c. Will only work for 7B model on device with enough memory.

```

### TODO:
```
[ ] LoRA updates to save on disk writes. Don't need to quantize anything, just avoid writing back all the parameters, only keep the part of it trainable.
[ ] AdamW support, save optimizer state as well
[ ] optimizations - prefetch the blackbox, save asyncronously, etc.
[ ] cleanup and explanation.
[ ] progress tracking for everything
[ ] different dtype to save intermediate tensors (help disk writes wear off, disk space usage and training time)
[ ] quantization
[ ] improve loading time as it is important for testing
[ ] for saving model, fix rope?
[ ] configurable weight tying
```

### References
* [llama](https://github.com/facebookresearch/llama)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [llama2.c](https://github.com/karpathy/llama2.c)
* [cubestat](https://github.com/okuvshynov/cubestat)
* [LoRA](https://arxiv.org/abs/2106.09685)
## slowllama

Fine-tune llama2 models (including 70B) on Apple macbooks. slowllama is not using any quantization (no lora, etc) and just offloads parts of model to/from SSD. This approach, while slow, can work for small-scale finetuning.
In contrast with training large models from scratch (unattainable) or inference (where we are likely to care about interactivity and tokens/sec), in finetune case we can still get something done if we allow it to run, say, overnight in batches of modest size.

### How does it work?
Most of the tests were done on MacMini M1 with 16Gb RAM with llama2-13b. 7B model will have one shard and 70B model will have 8, but other than that there's no special handling.

First, we need to be able to load a model which needs much more RAM than we have. We create model instance with all large modules' weights offloaded to SSD - for each of the transformer blocks, token embeddings and output linear layer. After that we load model checkpoints one by one, for each checkpoint iterate over all modules, update corresponding subset of its weights and save it back. 

Doing forward path is straighforward - we just load each model when we need, evaluate it and return the result. Backward pass is a little more tricky.

The way it's currently implemented is:
1. Do a forward pass the same way as above, while also saving inputs to each block.
2. Then, do a ~manual backward gradient propagation. We re-run each block again with the same input to build the internal graph again. Run real backward pass, update the weights for that block, save it back. Repeat. Important: we need to save and restore random number generation state. During training we use dropout, and randomly switched off neurons should be the same on both forward passes.

Limitations:
1. Stateless optimizer (SGD). We'll have to load/save AdamW state repeatedly as well to be able to use it. That's one of the next action items.
2. It is slow
3. SSD requirements - even for 13B model you'll need: 26-27Gb for original weights + 50Gb for intermediate weights (in float32) + same 26-27Gb if you want to save a single snapshot, so ~100-110Gb total. You are looking at something order of 500Gb for llama70.
4. no gradient accumulation, we just update weights as we go.


### How to setup

1. Install dependencies: torch, sentencepiece 
2. Follow instructions in [llama2](https://github.com/facebookresearch/llama) and download the models. It will download tokenizer model as well.
3. Clone that repo, we'll need it for tokenizer
4. To check that model loaded correctly we can test generating the sequence 

```
python test_gen.py path_to_folder [optional device]
```

For example, to run 7b model on macbook m1/m2:
```
python test_gen.py ../llama-2-7b mps
```

tokenizer.model should be put into the same directory as model itself. Running this relies on llama repo being at the same level as slowllama.
Example folder structure could look like:
```
/projects
    /llama-2-7b/...
    /llama-2-13b/...
    /llama-2-70b/...
    /llama/...     # <-- this is Meta repository
    /slowllama/... # <- this repo
```

5. Running a single backprop iteration test:

```
python test_backprop.py ../llama-2-7b data
```
This would run model on CPU and compare output and some weight subset after 1 step against data obtained through reference implementation (llama2.c). 

### How to finetune

1. Open finetune.py. It's a very simple file which loads small txt file with public domain text, tokenizes it and splits into train/validation set. There are some settings you could change here, like sequence length, batch size, learning rate, dropout rate, number of iterations. Change this if desired. Model path for input/output is hardcoded in that script as well, change accordingly. For the model saving, double-check that you use the expected number of shards (TODO: just reuse one from original model). 
2. run ```python finetune.py```

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
[x] just path to llama folder, no individual files
[x] make backprop work. Have to use larger device to test, no way to run locally. Actually, it worked but very slow.
[x] backprop: better handling of device, including backprop
[x] integration test
[x] export back to normal llama format.
[x] make dropout work. use get_rng_state to make forward/backward pass match.
[x] tokenizer + generation of something readable
[x] fix 'eval' mode for blackbox layers - dropout is not respected.
[?] training: fine-tune on a real dataset
[x] test on cuda
[x] rather than comparing to reference implementation save the output.
[x] pass learning rate around, not configure in 3 different places.
[x] check what exactly takes how much memory
[x] offload embeddings & output linear layer as well.
[x] get rid of dependency on llama.c on test 
[x] finetune + save/load + gen
[x] simpler loading, just load checkpoints one by one on CPU
[ ] progress tracking for everything
[x] larger llama (13B on mac, 70b on CUDA)
[x] try bfloat16 on cuda
[ ] cleanup and explanation. 

Later:
[ ] quantization: fp16, lora
[ ] AdamW support, save optimizer state as well
[ ] optimizations - prefetch the blackbox, save asyncronously, etc.
[ ] improve loading time as it is important for testing
[ ] for saving model, fix rope?
```

### References
* [llama2.c](https://github.com/karpathy/llama2.c)
* [llama](https://github.com/facebookresearch/llama)
* [cubestat](https://github.com/okuvshynov/cubestat)

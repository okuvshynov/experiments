### How to use

1. Install dependencies: torch, sentencepiece 
1. Follow instructions in [llama2](https://github.com/facebookresearch/llama) and download the models. It will download tokenizer model as well.
2. Clone that repo, we'll need it for tokenizer
3. To check that everything worked you can run 

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
    /llama/ # <-- this is Meta repository
    /slowllama/... # <- this repo
    /llama2.c # (https://github.com/karpathy/llama2.c), optional for 
```

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
[ ] AdamW support, save optimizer state as well.
[ ] lora quantization
[ ] optimizations - prefetch the blackbox, save asyncronously, measure utilization, etc.
[ ] improve loading time as it is important for testing
[ ] for saving model, fix rope?
```

### References
* [llama2.c](https://github.com/karpathy/llama2.c)
* [llama](https://github.com/facebookresearch/llama)
* [cubestat](https://github.com/okuvshynov/cubestat)

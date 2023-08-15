### testing 

Testing requires for now:
1. llama2.c (+ commend out weight-tying there)
2. llama from meta (repo with code)
3. llama2 weights + tokenizer

```
python split_model/test_backprop.py ../llama-2-7b/
python split_model/test_gen.py ../llama-2-7b/

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
[ ] optimization - prefetch the phantom, save asyncronously, measure utilization, etc.
[ ] fix 'eval' mode for phantom layers - dropout is not respected.
[ ] training: fine-tune on a real dataset
[ ] get rid of dependency on llama.c on test 
[ ] larger llama2 (15/70)?
[ ] training: test on large fast machine with cuda
```

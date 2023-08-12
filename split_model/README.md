### testing 

```
python split_model/forward_cmp.py ../llama-2-7b/
python split_model/backward_cmp.py ../llama-2-7b/
```


### TODO:
```
[x] just path to llama folder, no individual files
[x] make backprop work. Have to use larger device to test, no way to run locally. Actually, it worked but very slow.
[x] backprop: better handling of device, including backprop
[x] integration test
[x] export back to normal llama format.
[ ] make dropout work. use get_rng_state to make forward/backward pass match. https://pytorch.org/docs/stable/_modules/torch/random.html#manual_seed
[ ] training: fine-tune on a real dataset
[ ] optimization - prefetch the phantom, save asyncronously, measure utilization, etc.
[ ] larger llama2 (15/70)?
[ ] training: test on large fast machine with cuda
```

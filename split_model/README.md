### testing 

```
python split_model/forward_cmp.py ../llama-2-7b/
python split_model/backward_cmp.py ../llama-2-7b/
```


### TODO:
[x] just path to llama folder, no individual files
[x] make backprop work. Have to use larger device to test, no way to run locally. Actually, it worked but very slow.
[ ] backprop: create/clear data/ folder
[ ] training: fine-tune on a real dataset
[ ] export back to normal llama format.
[x] backprop: better handling of device, including backprop
[x] integration test
[ ] larger llama2 (15/70)?
[ ] backprop: communication/persistence protocol - where/how do we store and what do we override
[ ] training: test on large fast machine with cuda
[ ] optimization - prefetch the phantom, save asyncronously, measure utilization, etc.
[ ] stateful optimizer

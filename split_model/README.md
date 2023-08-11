### testing 

```
python split_model/forward_cmp.py ../llama-2-7b/
python split_model/backward_cmp.py ../llama-2-7b/
```


### TODO:
[x] just path to llama folder, no individual files
[x] make backprop work. Have to use larger device to test, no way to run locally. Actually, it worked but very slow.
[ ] backprop service
        [x] better handling of device, including backprop
        [ ] communication/persistence protocol - where/how do we store and what do we override
        [ ] create data/ folder
[ ] tests
        [ ] test on large fast machine with cuda
        [ ] fine-tune on a real dataset
        [?] integration test
        [ ] larger llama2 (15/70)?
[ ] export back to normal llama format.
[ ] some progress + measure ram usage internally
[ ] optimization - prefetch the phantom, save asyncronously
        [ ] measure time for each section
[ ] stateful optimizer

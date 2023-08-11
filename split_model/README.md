

### test loading llama2_7b

```
python split_model/manual_load.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
```

### check memory usage

```
python split_model/track_ram.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
```

### backwards pass

Manual check of old/new weights after 1 backwards pass based on 1 sample looks good. Need a real reproducible test. 

#### torch
before
```
Parameter containing:
tensor([[-0.0030, -0.0080, -0.0132,  ...,  0.0087, -0.0038, -0.0089],
        [-0.0175, -0.0095,  0.0095,  ...,  0.0054, -0.0236, -0.0060],
        [-0.0156, -0.0122,  0.0009,  ..., -0.0024,  0.0085, -0.0036],
        ...,
        [ 0.0049,  0.0113, -0.0217,  ...,  0.0118,  0.0148, -0.0085],
        [-0.0066,  0.0476,  0.0089,  ...,  0.0354, -0.0135,  0.0012],
        [-0.0073, -0.0220, -0.0153,  ..., -0.0121,  0.0222, -0.0072]],
       requires_grad=True)
```

after
```
Parameter containing:
tensor([[-0.0023, -0.0076, -0.0122,  ...,  0.0090, -0.0032, -0.0090],
        [-0.0184, -0.0098,  0.0077,  ...,  0.0048, -0.0245, -0.0055],
        [-0.0160, -0.0123, -0.0002,  ..., -0.0024,  0.0078, -0.0037],
        ...,
        [ 0.0058,  0.0119, -0.0205,  ...,  0.0124,  0.0153, -0.0093],
        [-0.0054,  0.0485,  0.0103,  ...,  0.0364, -0.0132, -0.0001],
        [-0.0078, -0.0221, -0.0164,  ..., -0.0124,  0.0215, -0.0071]],
       requires_grad=True)
```


#### Phantom:
before 
```
Parameter containing:
tensor([[-0.0030, -0.0080, -0.0132,  ...,  0.0087, -0.0038, -0.0089],
        [-0.0175, -0.0095,  0.0095,  ...,  0.0054, -0.0236, -0.0060],
        [-0.0156, -0.0122,  0.0009,  ..., -0.0024,  0.0085, -0.0036],
        ...,
        [ 0.0049,  0.0113, -0.0217,  ...,  0.0118,  0.0148, -0.0085],
        [-0.0066,  0.0476,  0.0089,  ...,  0.0354, -0.0135,  0.0012],
        [-0.0073, -0.0220, -0.0153,  ..., -0.0121,  0.0222, -0.0072]],
       requires_grad=True)
```

after
```
Parameter containing:
tensor([[-0.0023, -0.0076, -0.0122,  ...,  0.0090, -0.0032, -0.0090],
        [-0.0184, -0.0098,  0.0077,  ...,  0.0048, -0.0245, -0.0055],
        [-0.0160, -0.0123, -0.0002,  ..., -0.0024,  0.0078, -0.0037],
        ...,
        [ 0.0058,  0.0119, -0.0205,  ...,  0.0124,  0.0153, -0.0093],
        [-0.0054,  0.0485,  0.0103,  ...,  0.0364, -0.0132, -0.0001],
        [-0.0078, -0.0221, -0.0164,  ..., -0.0124,  0.0215, -0.0071]],
       requires_grad=True)
```

### TODO:
[x] just path to llama folder, no individual files
[x] make backprop work. Have to use larger device to test, no way to run locally. Actually, it worked but very slow.
[ ] test on large fast machine with cuda
[ ] better handling of device, including backprop
[ ] some progress + measure ram usage internally
[ ] create data/ folder
[ ] fine-tune on a real dataset
[ ] larger llama2 (15/70)?
[ ] file path protocol - where/how do we store and what do we override
[ ] optimization - prefetch the phantom, save asyncronously
[ ] integration test
[ ] stateful optimizer
[ ] export back to normal llama format.
## some basic optimizations 

All of these were done on Apple M1 with 16Gb RAM and 256Gb SSD.

### Step 1 - measure

Observe current state when running [benchmark](backprop_bm.py):

![cubestat utilization](static/backprop_0.png)

```
Loaded 292 module metadata
Created blank model
processing transformer blocks ................................ DONE
populated all weights to model
loaded phantom model in 88.40827322006226 seconds
forward pass in 69.51894903182983 seconds
backward pass in 243.84246110916138 seconds
forward pass in 74.14301109313965 seconds
backward pass in 244.7289023399353 seconds
forward pass in 74.9399299621582 seconds
```

### No re-importing torch

Simplest thing first, we don't need to start separate process every time and thus re-importing torch again and again.

https://github.com/okuvshynov/experiments/commit/a12dcd14f18b378f2764e00954434ab84a3836db changes this to be a process starting once.

![cubestat utilization](static/backprop_import.png)

As we can see, time is down considerably and gaps between 100% utilization in GPU are shorter. 

```
% python split_model/backprop_bm.py ../llama-2-7b
Loaded 292 module metadata
Created blank model
processing transformer blocks ................................ DONE
populated all weights to model
loaded phantom model in 89.22977995872498 seconds
forward pass in 69.22715783119202 seconds
backward pass in 203.22368097305298 seconds
```


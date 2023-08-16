## some basic optimizations 

All of these were done on Apple M1 with 16Gb RAM and 256Gb SSD.

In this case we have multiple components we can look at, including utilization for CPU, GPU, RAM, disk, etc. 

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
Loaded 292 module metadata
Created blank model
processing transformer blocks ................................ DONE
populated all weights to model
loaded phantom model in 102.21748113632202 seconds
forward pass in 69.36935567855835 seconds
backward pass in 207.33418202400208 seconds
forward pass in 77.83041286468506 seconds
backward pass in 211.1048882007599 seconds
forward pass in 77.09704279899597 seconds
backward pass in 204.30037093162537 seconds
```

### prefetch and async weights save

Looking at utilization plot at higher resolution (100ms time step) we can see what's going on more clearly:

![cubestat utilization](static/backprop_hires.png)

To improve on this gap, we can prefetch next block from disk in advance and save the current module weights asyncronously.
This, however, will result in higher memory usage as we'll have to keep multiple transformer modules in RAM simultaneuously.

Another way to shorten these gaps would be to optimize the loading process without introducing parallelism - for example, just update the 
weights rather than saving entire transformer block.


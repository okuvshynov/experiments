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

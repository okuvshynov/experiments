# llama duo - asyncronous speculative decoding for llama3. 

llama duo is an attempt to make a simple speculative decoding work in parallel with the main model.
Not every hardware/model combination would benefit from such setup, here are some examples where it worked reasonably well:
1. Llama3-8B @ fp16 running on Apple M2 24Gb laptop and Llama3-8B@Q4 running Apple M1 16Gb mac mini.
2. Llama3-70B @ Q8 running on M2 Ultra GPU and Llama3-8B @Q3 running on same M2 Ultra CPUs.

The benefits of doing it:
1. You can run a decent speculative model without incurring large latency cost for synchronous speculative model evaluation.
2. You can fit a decent speculative model into memory, if you have an extra device.

## Dependencies

1. llama.cpp
2. nlohmann/json
3. cpp-httplib

For the chat.py, needs python and requests.

## Installation

1. clone this repo
2. ```mkdir _build && cd _build```
3. ```cmake ..```
4. ```make -j 4```
5. ```pip install requests```

After this step you should have two binaries built: ```lead``` and ```back```. 

## Local example

Here we run it on single M2 Ultra.

Start lead with Llama3-70B@Q8 model with all layers on GPU and default settings for interface/port (0.0.0.0:5555):

```./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99```

Start back with Llama3-8B@Q4 model on 16 CPU threads. It looks for lead service on localhost:5555 by default.

```./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf --n-gpu-layers 0 --threads 16```

Start basic chat command-line interface (also defaults to localhost:5555):

```python chat.py```

In chat window ask the model something: 

```You: Illustrate the difference between concurrency and parallelism in python.```

What we should observe:

1. ```lead``` service should start printing out the generated tokens, highlighing accepted tokens in green.

<img width="623" alt="Screenshot 2024-05-14 at 10 10 05 AM" src="https://github.com/okuvshynov/experiments/assets/661042/40454bf7-78e2-46f1-b770-661a55e6e05a">

2. ```back``` would print some debug info.

3. After the generation is complete, the response would be returned to chat.


```lead``` would print out some timing info:


```
I: encoded  105 tokens in    3.108 seconds, speed:   33.786 t/s
...
I: decoded  784 tokens in   75.159 seconds, speed:   10.431 t/s
I: total generation time: 78.2696
```

Note that ```back``` service is optional - we can turn it off, run the main model as before:

```./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99```

```python chat.py```

In chat window ask the same question: 

```You: Illustrate the difference between concurrency and parallelism in python.```

And observe the same output.

```
I: encoded  105 tokens in    2.699 seconds, speed:   38.908 t/s
...
I: decoded  784 tokens in   92.639 seconds, speed:    8.463 t/s
I: total generation time: 95.3407
```

As we can see, it is slower.


We can also start/stop/simulate non-availability/failure for ```back``` service. As in previous example, start main model and chat:

```./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99```

```python chat.py```

In chat window ask the model the same question: 

```You: Illustrate the difference between concurrency and parallelism in python.```

At some moment during generation start the ```back``` service:

```./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf --n-gpu-layers 0 --threads 16```

```back``` service would catch up with ```lead``` by processing input prompt + the tokens generated to this point and start speculating.
The performance would be somewhere in between the two runs above

```
I: encoded  105 tokens in    2.765 seconds, speed:   37.969 t/s
...
I: decoded  784 tokens in   82.254 seconds, speed:    9.568 t/s
I: total generation time: 85.0213
```

We can also kill the back service sometime in the middle of query processing, start it again, etc.


## Distributed example

On M2 Macbook with 24 Gb memory start ```lead``` service with full fp16 precision 8B model:

```
./lead -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-fp16.gguf -ngl 99
```

On M1 Mini with 16Gb memory start ```back``` service and specify the ```lead``` host:

```
./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct.Q3_K_M.gguf --host 169.254.226.241 -ngl 99
```

Both of these services will run on GPUs. The model they run is essentially the same, except smaller and slower machine runs more aggressively quantized version.

Now on the macbook start the chat and ask the same question:

```python chat.py```

```You: Illustrate the difference between concurrency and parallelism in python.```

```
...
I: decoded  737 tokens in   81.129 seconds, speed:    9.084 t/s
I: total generation time: 100.386
```

Running same model without speculation would be much slower:

```
..
I: decoded  737 tokens in  222.631 seconds, speed:    3.306 t/s
I: total generation time: 224.635
```


On the other hand, if ```lead``` service would run a smaller model (like llama3-8B @ Q8) there would not be much benefit in distributed speculation.

## How it works

It is simple linear speculation, except it is generated in parallel with main model and reconciled after each main model token generation.

We can think of three separate sequences:
1. local sequence on ```lead``` -- this is ground truth, which will be equivalent to main model producing tokens one by one. Let's call this sequence L.
2. local sequence on ```back``` -- this is the speculated sequence which we work on in parallel. Let's call this sequence B.
3. shared speculation sequence on ```lead``` -- it serves as a communication channel between ```lead``` and ```back``` models. Let's call this sequence S.

Within S, tokens might be in the following states:
1. approved - this token was generated by main model and is not going to change
2. not_rejected - this token was produced by speculation model, but we don't know yet if it will be approved or not.
3. rejected - speculation model produced it, but main model generated different sequence after that, so we need to remove it.

Let's look at the following example:

consider prompt 'The quick brown'. All sequences L, B and S are initialized with it.

```lead``` and ```back``` start working on it in parallel. All operations involving S are guarded with mutex so that lead and back would not modify it simultaneously. Let's consider the following event sequence.

```
L = [the, quick, brown]
B = [the, quick, brown]
S = [the, quick, brown]

back produces 'fox'.
L = [the, quick, brown]
B = [the, quick, brown, fox]
S = [the, quick, brown]

back calls lead and compares B with S. 'fox' and appended to the S in 'not_rejected' state. 
L = [the, quick, brown]
B = [the, quick, brown, fox]
S = [the, quick, brown, fox]

back produces 'jumps'.
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps]
S = [the, quick, brown, fox]

back calls lead and compares B with S. 'jumps' is appended to S in 'not_rejected' state. 
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps]
S = [the, quick, brown, fox, jumps]

back produces 'into'.
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps, into]
S = [the, quick, brown, fox, jumps]

back calls lead and compares B with S. 'into' is appended to S in 'not_rejected' state. 
L = [the, quick, brown]
B = [the, quick, brown, fox, jumps, into]
S = [the, quick, brown, fox, jumps, into]

lead produces 'fox'. 'fox' is appended to L.
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into]
S = [the, quick, brown, fox, jumps, into]

lead compares L with S. As 'fox' matches, it is marked is approved, 'jumps into' stays not_rejected, main model starts working on input of 3 tokens 'fox jumps into'.
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into]
S = [the, quick, brown, fox, jumps, into]

back produces 'the'.
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the]
S = [the, quick, brown, fox, jumps, into]

back calls lead and compares B with S. 'the' is appended to S in 'not_rejected' state.
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the]
S = [the, quick, brown, fox, jumps, into, the]

back produces 'big'.
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [the, quick, brown, fox, jumps, into, the]

back calls lead and compares B with S. 'big' is appended to S in 'not_rejected' state.
L = [the, quick, brown, fox]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [the, quick, brown, fox, jumps, into, the, big]

lead produces 'jumps over the'. First, we need to compare the output with input (in this case, 'fox jumps into'). As 'jumps' matches, but 'over' != 'into', we accept 'jumps over' and append it to L. We cannot accept 'the', because it was produced as an continuation to the sequence 'the quick brown fox jumps into', and we now know that 'into' was wrong.
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [the, quick, brown, fox, jumps, into, the, big]

lead compares L with S. We mark 'into the big' as rejected, remove them from the sequence S and assign S := L. ```lead``` works on a single input 'over'.
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, into, the, big]
S = [the, quick, brown, fox, jumps, over]

back produces 'puddle'.
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, into, the, big, puddle]
S = [the, quick, brown, fox, jumps, over]

back calls lead and compares B with S. We see a mismatch, append nothing to S, and assign B := S.
L = [the, quick, brown, fox, jumps, over]
B = [the, quick, brown, fox, jumps, over]
S = [the, quick, brown, fox, jumps, over]

```

The actual implementation is a little more complicated because communication between lead and back involves passing 'deltas' rather than entire sequences - otherwise we'd end up with considerable network latency for large contexts.


## Comparison with synchronous speculative decoding 

## Configuration options

## Next steps
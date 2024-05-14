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
2. mkdir _build && cd _build
3. cmake ..
4. make -j 4
5. pip install requests

## Local example

Here we run it on single M2 Ultra.

After this step you should have two binaries built: lead and back. 

Start lead with Llama3-70B@Q8 model with all layers on GPU and default settings for interface/port (0.0.0.0:5555):

```./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99```

Start back with Llama3-8B@Q4 model on 16 CPU threads. It looks for lead service on localhost:5555 by default.
```./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf --n-gpu-layers 0 --threads 16```

Start basic chat command-line interface (also defaults to localhost:5555):
```python chat.py```

In chat window ask the model something: 
```You: Illustrate the difference between concurrency and parallelism in python.```

What we should observe:
1. lead service should start printing out the generated tokens, highlighing accepted tokens.

<img width="623" alt="Screenshot 2024-05-14 at 10 10 05 AM" src="https://github.com/okuvshynov/experiments/assets/661042/40454bf7-78e2-46f1-b770-661a55e6e05a">

2. back would print some debug info:

3. After the generation is complete, the response would be returned to chat.

lead would print out some timing info:

```
I: encoded  105 tokens in    3.108 seconds, speed:   33.786 t/s
...
I: decoded  784 tokens in   75.159 seconds, speed:   10.431 t/s
I: total generation time: 78.2696
```

Note that 'back' service is optional - we can turn it off, run the main model as before:
```./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99```
```python chat.py```

In chat window ask the model the same question: 
```You: Illustrate the difference between concurrency and parallelism in python.```

And observe a slower, but same output.
```
I: encoded  105 tokens in    2.699 seconds, speed:   38.908 t/s
...
I: decoded  784 tokens in   92.639 seconds, speed:    8.463 t/s
I: total generation time: 95.3407
```
As we can see, it is slightly slower.


We can also start/stop/simulate non-availability for back service. As in previous example, start main model and chat:
```./lead -m ../../../llms/gguf/Meta-Llama-3-70B-Instruct-v2.Q8_0-00001-of-00003.gguf --n-gpu-layers 99```
```python chat.py```
In chat window ask the model the same question: 
```You: Illustrate the difference between concurrency and parallelism in python.```

At some moment during generation start the back service:
```./back -m ../../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q4_0.gguf --n-gpu-layers 0 --threads 16```

back service would catch up with main by processing input prompt + the tokens generated to this point and start speculating.
The performance would be somewhere in between 

```
I: encoded  105 tokens in    2.765 seconds, speed:   37.969 t/s
...
I: decoded  784 tokens in   82.254 seconds, speed:    9.568 t/s
I: total generation time: 85.0213
```

We can also kill the back service sometime in the middle of query processing.


## Distributed example



## How it works

## Comparison with synchronous speculative decoding 

## Configuration options

## Next steps

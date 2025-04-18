# local LLM summarization MCP server.

Scenario:

I'd like to work on a somewhat large project using claude code or claude desktop app. Summarizing entire project with claude models would hit usage limit for my account, but proceeding without summary often results in a lot of exploration (and hitting the limits anyway).

Instead, we'll use local llm to summarize files and directories, and instruct claude to use that tool by passing the paths to files, not files themselves. This way claude will not have to read entire project and can get summaries cheaper. At the same time strong model (like sonnet 3.7) can be used to identify 'what to summarize'.

## Usage with claude code:

### Install claude code: 

https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview

### Install uv if you don't have it

```curl -LsSf https://astral.sh/uv/install.sh | sh```


### Run open-ai compatible local llm server

I used [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)

```./llama.cpp/build/bin/llama-server -m path/to/your/local/model.gguf -n 8192 -c 65536 --host 0.0.0.0```

### Get this repo 

```git clone https://github.com/okuvshynov/experiments.git```

### Configure claude code to use the mcp server

```claude mcp add summarize -s user -- uv "--directory" /Absolute/path/to/experiments/mcps/summarize run summarize.py```. 

In this case we add it globally, not per project (-s flag)

### Now within claude code we can instruct it like this

(Using https://github.com/okuvshynov/cubestat as example)

```
> Use summarize tool to get the summaries for metrics implementations. Do not read the content of the file yourself, get the summary using the tool

  summarize:summarize   (file_paths: ["cubestat/metrics/base_metric.py","cubestat/metrics/cpu.py","cubestat/metrics/memory.py","cubestat/metrics/disk.py","cubestat/metrics/gpu.py","cubestat/…e
  (MCP)                 trics/network.py","cubestat/metrics/accel.py","cubestat/metrics/power.py","cubestat/metrics/swap.py","cubestat/metrics/mock.py"], root:
                        "/Users/oleksandr/projects/cubestat")
...

Here are the summaries of the metric implementations:
  - base_metric.py: Abstract base class with abstract methods subclasses must implement and some default implementations.
  - cpu.py: CPU usage monitoring with platform-specific implementations, display modes, and command line arguments.
  - memory.py: Memory usage monitoring with platform-specific implementations, display modes, and command line arguments.
  - disk.py: Disk I/O monitoring with platform-specific implementations, display modes, and command line arguments.
  - gpu.py: GPU monitoring with platform-specific implementations, display modes, and command line arguments.
  - network.py: Network I/O monitoring with platform-specific implementations, display modes, and command line arguments.
  - accel.py: Accelerator (ANE) monitoring for macOS only with display modes and command line arguments.
  - power.py: Power usage monitoring for macOS only with display modes and command line arguments.
  - swap.py: Swap memory monitoring with platform-specific implementations, display modes, and command line arguments.
  - mock.py: Mock metric class for testing with a simple platform-agnostic implementation.

```

Observe very low cost on claude side and 10k tokens on local llm.

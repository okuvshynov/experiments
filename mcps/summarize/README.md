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

### define custom claude command to your project

We use https://github.com/okuvshynov/cubestat.

```git clone https://github.com/okuvshynov/cubestat.git```

```mkdir -p cubestat/.claude/commands && cp experiments/mcps/commands/initref.md cubestat/.claude/commands```

Now, we can use initref command from within claude code app, which will:
1. Pass files for summarization to our local LLM
2. Use the summaries to generate documentation, saving on tokens/limits
3. Result is https://github.com/okuvshynov/cubestat/commit/cd482d964ccbc2449d56e99a87f0172df412d489


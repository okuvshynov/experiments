local LLM summarization MCP server.

Use case:

I'd like to work on a somewhat large project using claude code or claude desktop app. Summarizing entire project with claude models would hit usage limit for my account, but proceeding without summary often results in a lot of exploration (and hitting the limits anyway).

Instead, we'll use local llm to summarize files and directories, and instruct claude to use that tool (passing the path to files, not files themselves). This way claude will not have to read entire project.

Usage with claude code:


```

```

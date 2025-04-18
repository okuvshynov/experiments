from typing import Any, List
import httpx
import os.path
import sys
import asyncio
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("summarize")

prompt = """
For each file provided, write a brief summary. Include the connections between files if you have identified them.
"""

async def token_count(content: str) -> int:
    headers = {
        "Content-Type": "application/json"
    }
    url = "http://localhost:8080/tokenize"
    request = {
        "content": content
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=request, timeout=30.0)
            response.raise_for_status()  # Raise exception for non-200 responses
            result = response.json()
            return len(result["tokens"])
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return 0

async def summarize_impl(content: str) -> str | None:
    headers = {
        "Content-Type": "application/json"
    }
    url = "http://localhost:8080/v1/chat/completions"
    request = {
        "messages": [{"role": "user", "content": f"{prompt}\n{content}"}]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=request, timeout=30.0)
            response.raise_for_status()  # Raise exception for non-200 responses
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    return result["choices"][0]["message"]["content"]
            return f"Invalid response format: {result}"
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg

@mcp.tool()
async def summarize(file_paths: List[str], root: str) -> str:
    """Summarize the content of multiple files

    Args:
        file_paths: List of file paths to summarize (relative to root)
        root: Root directory where files are located
    """
    combined_content = ""
    for rel_path in file_paths:
        try:
            full_path = os.path.join(root, rel_path)
            with open(full_path, 'r') as file:
                content = file.read()
                combined_content += f"{rel_path}:\n{content}\n\n"
        except Exception as e:
            combined_content += f"{rel_path}:\nError reading file: {str(e)}\n\n"
            
    tc = await token_count(combined_content)
    print(f"Token count = {tc}")
    summary = await summarize_impl(combined_content)
    return summary if summary else "Failed to generate summary"

async def test_summarize():
    """Test function that summarizes the current script file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file = os.path.basename(__file__)

    result = await summarize([current_file], current_dir)
    print(f"Summary of {current_file}:\n{result}")

if __name__ == "__main__":
    if "--test" in sys.argv:
        # Run the test function
        asyncio.run(test_summarize())
    else:
        # Initialize and run the server
        mcp.run(transport='stdio')

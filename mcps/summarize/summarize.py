from typing import Any, List
import httpx
import os.path
import sys
import asyncio
import logging
import argparse
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('summarize')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Summarize files content')
parser.add_argument('--max-tokens', type=int, default=2**14, 
                    help='Maximum tokens per batch')
parser.add_argument('--base-url', type=str, default='http://localhost:8080',
                    help='Base URL for API endpoints (default: http://localhost:8080)')
parser.add_argument('--test', action='store_true', help='Run test function')
args, unknown = parser.parse_known_args()

# Initialize FastMCP server
mcp = FastMCP("summarize")

prompt = """
For each file provided, write a brief summary. Include the connections between files if you have identified them.
"""

MAX_TOKENS_PER_BATCH = args.max_tokens
BASE_URL = args.base_url

logger.info(f"Configured with MAX_TOKENS_PER_BATCH={MAX_TOKENS_PER_BATCH}, BASE_URL={BASE_URL}")

async def token_count(content: str) -> int:
    headers = {
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/tokenize"
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
        error_msg = f"Error calculating token count: {str(e)}"
        logger.error(error_msg)
        return 0

async def summarize_impl(content: str) -> str | None:
    headers = {
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/v1/chat/completions"
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
        error_msg = f"Error during summarization: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
async def summarize(file_paths: List[str], root: str) -> str:
    """Summarize the content of multiple files and relationships between them

    Args:
        file_paths: List of file paths to summarize (relative to root)
        root: Root directory where files are located
    """
    
    # First, load all files and count tokens for each
    file_contents = []
    file_token_counts = []
    
    for rel_path in file_paths:
        try:
            full_path = os.path.join(root, rel_path)
            with open(full_path, 'r') as file:
                content = f"{rel_path}:\n{file.read()}\n\n"
                file_contents.append(content)
                tokens = await token_count(content)
                file_token_counts.append(tokens)
                logger.info(f"File {rel_path}: {tokens} tokens")
        except Exception as e:
            error_content = f"{rel_path}:\nError reading file: {str(e)}\n\n"
            file_contents.append(error_content)
            file_token_counts.append(await token_count(error_content))
    
    # Split files into batches that don't exceed MAX_TOKENS_PER_BATCH
    batches = []
    current_batch = []
    current_batch_tokens = 0
    
    for i, (content, tokens) in enumerate(zip(file_contents, file_token_counts)):
        # If a single file exceeds the token limit, we'll still include it in its own batch
        if current_batch_tokens + tokens > MAX_TOKENS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_batch_tokens = 0
        
        current_batch.append(content)
        current_batch_tokens += tokens
    
    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    logger.info(f"Split into {len(batches)} batches")
    
    # Process each batch and concatenate results
    all_summaries = []
    for i, batch in enumerate(batches):
        combined_content = "".join(batch)
        logger.info(f"Processing batch {i+1}/{len(batches)}")
        batch_summary = await summarize_impl(combined_content)
        if batch_summary:
            all_summaries.append(batch_summary)
        else:
            all_summaries.append(f"Failed to generate summary for batch {i+1}")
    
    # Combine all summaries
    final_summary = "\n\n".join(all_summaries)
    return final_summary

async def test_summarize():
    """Test function that summarizes the current script file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_file = os.path.basename(__file__)

    result = await summarize([current_file], current_dir)
    logger.info(f"Summary of {current_file}:\n{result}")

if __name__ == "__main__":
    if args.test:
        # Run the test function
        asyncio.run(test_summarize())
    else:
        # Initialize and run the server
        mcp.run(transport='stdio')

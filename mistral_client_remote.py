#!/usr/bin/env python
import os
import pdb
import json
import asyncio
import traceback
from pprint import pprint

from mistralai import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams
from pathlib import Path

from mistralai.types import BaseModel
from typing import List, Optional, Union, Any, Dict

from dotenv import load_dotenv

color_blue = "\033[94m"
color_reset = "\033[0m"

load_dotenv()

# Set the current working directory and model to use
cwd = Path(__file__).parent
print(f"{color_blue}cwd = {cwd}{color_reset}")
MODEL = "mistral-medium-latest"
MODEL = "mistral-small-latest"
print(f"{color_blue}MODEL = {MODEL}{color_reset}")

# Generic structured output model for flexible responses
class GenericResponse(BaseModel):
    result: str
    status: bool

class MCPClient:
    def __init__(self, server_urls: List[str] = [], api_key: str = None):
        """Initialize the MCP client.
        
        Args:
            server_urls: URL(s) of the MCP server(s). Can be a single URL string or list of URLs
            api_key: Mistral API key (if None, will try to get from environment)
        """
        self.server_urls = server_urls
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass it directly.")
        
        self.client = Mistral(self.api_key)
        
        print(f"{color_blue}MCP Client initialized with servers: {self.server_urls}{color_reset}")
    
    async def query_single_server(self, user_input: str, server_url: str, server_index: int = 0) -> dict:
        """Send a query to a single MCP server and get a response.
        
        Args:
            user_input: The user's query
            server_url: URL of the specific server to query
            server_index: Index of the server (for identification)
            
        Returns:
            Dictionary containing the response with server information
        """
        print(f"{color_blue}\nProcessing query on server {server_index + 1} ({server_url}): {user_input}{color_reset}")
        
        # Use provided output format or default to generic
        response_format = GenericResponse
        
        try:
            # Create a new MCP client for each request to avoid connection reuse issues
            mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=server_url, timeout=100))
            
            # Create run context
            async with RunContext(
                model=MODEL,
                output_format=response_format,
            ) as run_ctx:
                # Register the MCP client
                await run_ctx.register_mcp_client(mcp_client=mcp_client)
                
                # Run the query
                run_result = await self.client.beta.conversations.run_async(
                    run_ctx=run_ctx,
                    inputs=user_input,
                )

                # Extract the response
                response = dict(run_result.output_as_model)
                print(f"{color_blue}Response received:")
                pprint(response)
                print(color_reset)
                
                return {
                    "success": True,
                    "response": response,
                    "conversation_id": run_result.conversation_id,
                    "error": None,
                    "server_url": server_url,
                    "server_index": server_index,
                }
                
        except Exception as e:
            traceback.print_exc()
            error_response = {
                "success": False,
                "response": None,
                "conversation_id": None,
                "error": str(e),
                "server_url": server_url,
                "server_index": server_index,
            }
            print(f"{color_blue}Error occurred on server {server_index + 1}: {e}{color_reset}")
            return error_response

    async def query_parallel(self, user_input: str, server_urls: List[str] = []) -> List[dict]:
        """Send a query to multiple MCP servers in parallel and return merged results.
        
        Args:
            user_input: The user's query
            server_urls: Optional list of specific server URLs to query. 
                        If None, uses all configured servers.
            
        Returns:
            List of dictionaries containing responses from all servers
        """
        if len(server_urls) < 2:
            raise ValueError("At least 2 servers are required for parallel queries")
        
        print(f"{color_blue}\n{'='*60}{color_reset}")
        print(f"{color_blue}Starting parallel queries to {len(server_urls)} servers{color_reset}")
        print(f"{color_blue}Query: {user_input}{color_reset}")
        print(f"{color_blue}{'='*60}{color_reset}")
        
        # Create tasks for parallel execution
        tasks = []
        for i, server_url in enumerate(server_urls):
            task = self.query_single_server(user_input, server_url, i)
            tasks.append(task)
        
        # Execute all queries in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle exceptions that occurred during execution
                    error_result = {
                        "success": False,
                        "response": None,
                        "conversation_id": None,
                        "error": str(result),
                        "server_url": server_urls[i],
                        "server_index": i,
                    }
                    processed_results.append(error_result)
                    print(f"{color_blue}Exception on server {i + 1}: {result}{color_reset}")
                else:
                    processed_results.append(result)
            
            # Print summary
            successful_results = [r for r in processed_results if r["success"]]
            failed_results = [r for r in processed_results if not r["success"]]
            
            print(f"{color_blue}\n{'='*60}{color_reset}")
            print(f"{color_blue}Parallel Query Summary:{color_reset}")
            print(f"{color_blue}✅ Successful: {len(successful_results)}/{len(processed_results)}{color_reset}")
            print(f"{color_blue}❌ Failed: {len(failed_results)}/{len(processed_results)}{color_reset}")
            print(f"{color_blue}{'='*60}{color_reset}")
            
            return processed_results
            
        except Exception as e:
            print(f"{color_blue}Error in parallel execution: {e}{color_reset}")
            traceback.print_exc()
            return []

def get_user_input() -> str:
    """Get user input from command line."""
    print(f"{color_blue}\n" + "="*60 + color_reset)
    print(f"{color_blue}MCP Client - Enter your query (or 'quit' to exit){color_reset}")
    print(f"{color_blue}" + "="*60 + color_reset)
    
    query = input("\nYour query: ").strip()
    return query

async def interactive_mode():
    """Run the client in interactive mode."""
    print(f"{color_blue}Starting interactive MCP client...{color_reset}")
    
    # Configure multiple servers
    default_servers = [
        "http://127.0.0.1:7860/gradio_api/mcp/sse",
        "http://127.0.0.1:7862/gradio_api/mcp/sse"  # Second server on different port
    ]
    
    try:
        # Initialize the client with multiple servers
        client = MCPClient(server_urls=default_servers)
        
        while True:
            user_query = get_user_input()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print(f"{color_blue}Goodbye!{color_reset}")
                break
            
            if not user_query:
                print(f"{color_blue}Please enter a valid query.{color_reset}")
                continue
            
            # Ask user which mode to use
            print(f"{color_blue}Choose mode:{color_reset}")
            print(f"{color_blue}1. Single server (first server only){color_reset}")
            print(f"{color_blue}2. Parallel query (all configured servers){color_reset}")
            
            mode = input("Enter mode (1/2) [default: 1]: ").strip()
            if not mode:
                mode = "1"
            
            try:
                if mode == "1":
                    # Single server query
                    # result = await client.query_single_server(user_query, client.server_urls[0], 0)
                    result = await client.query_single_server(user_query, client.server_urls[1], 1)
                    if result["success"]:
                        print(f"{color_blue}\n✅ Query processed successfully!{color_reset}")
                        print(f"{color_blue}Conversation ID: {result.get('conversation_id', 'N/A')}{color_reset}")
                        pprint(result["response"])
                    else:
                        print(f"{color_blue}\n❌ Error: {result['error']}{color_reset}")
                
                elif mode == "2":
                    # Parallel query to all servers
                    results = await client.query_parallel(user_query, client.server_urls)
                    print(f"{color_blue}\n📊 Results from {len(results)} servers:{color_reset}")
                    for i, result in enumerate(results):
                        print(f"{color_blue}\n--- Server {i + 1} ({result.get('server_url', 'unknown')}) ---{color_reset}")
                        if result["success"]:
                            print(f"{color_blue}✅ Success - Conversation ID: {result.get('conversation_id', 'N/A')}{color_reset}")
                            pprint(result["response"])
                        else:
                            print(f"{color_blue}❌ Error: {result['error']}{color_reset}")

                else:
                    print(f"{color_blue}Invalid mode selected. Please choose 1, 2, or 3.{color_reset}")
                    
            except Exception as e:
                print(f"{color_blue}\n❌ Execution error: {e}{color_reset}")
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print(f"{color_blue}\n\nOperation cancelled by user.{color_reset}")
    except Exception as e:
        print(f"{color_blue}\nError initializing client: {e}{color_reset}")

async def main():
    """Main entry point."""
    import sys

    # Interactive mode
    await interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())

"""
Usage Examples:

# Interactive mode with multiple servers
clear; python mistral_client_remote.py

# Make sure both MCP servers are running first:
clear;
if :; then
    python MCP_server_grab.py --port 7860 &
    python MCP_server_gojek.py --port 7862 &
fi
jobs
wait

Example queries:
    I want to go from Orchard Road to MBS
    I want to go from Woodlands to CBD
    Order pizza from Pizza Hut to my home
    Order pizza from KFC to office
"""

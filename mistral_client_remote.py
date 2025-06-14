#!/usr/bin/env python
import os
import pdb
import json
import asyncio
import traceback
from pprint import pprint

import gradio as gr
from mistralai import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams
from pathlib import Path

from mistralai.types import BaseModel
from typing import List, Optional, Union, Any, Dict

from dotenv import load_dotenv

color_blue = "\033[94m"
color_yellow = "\033[93m"
color_red = "\033[91m"
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
    result_dict: Dict[str, Any]
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
        
        response_output_dict = {}
        try:
            # Create a new MCP client for each request to avoid connection reuse issues
            print(f"{color_yellow}Creating new MCP client for server {server_index + 1} ({server_url}){color_reset}")
            mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=server_url, timeout=100))
            
            # Create run context
            print(f"{color_yellow}# Create run context{color_reset}")
            async with RunContext(
                model=MODEL,
                output_format=response_format,
            ) as run_ctx:
                # Register the MCP client
                print(f"{color_yellow}# Register the MCP client{color_reset}")
                await run_ctx.register_mcp_client(mcp_client=mcp_client)
                
                # Run the query
                print(f"{color_yellow}# Run the query{color_reset}")
                run_result = await self.client.beta.conversations.run_async(
                    run_ctx=run_ctx,
                    inputs=user_input,
                )
                print(f"{color_yellow}run_result = {run_result}{color_reset}")

                # Extract the response
                print(f"{color_yellow}# Extract the response{color_reset}")
                response = dict(run_result.output_as_model)
                print(f"{color_blue}response =")
                pprint(response)
                print(color_reset)
                
                response_output_dict = {
                    "success": True,
                    "response": response,
                    "conversation_id": run_result.conversation_id,
                    "error": None,
                    "server_url": server_url,
                    "server_index": server_index,
                }

        except Exception as e:
            traceback.print_exc()
            response_output_dict = {
                "success": False,
                "response": None,
                "conversation_id": None,
                "error": str(e),
                "server_url": server_url,
                "server_index": server_index,
            }
            print(f"{color_blue}Error occurred on server {server_index + 1}: {e}{color_reset}")

        return response_output_dict

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

# Global client instance
client = None

def initialize_client():
    """Initialize the global MCP client."""
    global client
    if client is None:
        print(f"{color_yellow}Initializing MCP client...{color_reset}")
        default_servers = [
            "http://127.0.0.1:7860/gradio_api/mcp/sse",
            "http://127.0.0.1:7862/gradio_api/mcp/sse"
        ]
        try:
            client = MCPClient(server_urls=default_servers)
            return True, "Client initialized successfully!"
        except Exception as e:
            return False, f"Error initializing client: {str(e)}"
    print(f"{color_yellow}Client already initialized{color_reset}")
    return True, "Client already initialized"

async def process_query(user_query: str, query_mode: str):
    """Process a user query through the MCP client."""
    global client
    
    if not user_query.strip():
        return {
            "error": "Please enter a valid query",
            "results": None,
            "summary": None
        }
    
    # Initialize client if needed
    success, message = initialize_client()
    if not success:
        return {
            "error": message,
            "results": None,
            "summary": None
        }
    
    try:
        if query_mode == "Single Server (Grab)":
            # Query first server (Grab)
            print(f"{color_yellow}Querying first server (Grab){color_reset}")
            result = await client.query_single_server(user_query, client.server_urls[0], 0)
            
            if result["success"]:
                summary = {
                    "mode": "Single Server",
                    "server": "Grab (7860)",
                    "status": "✅ Success",
                    "conversation_id": result.get('conversation_id', 'N/A'),
                }
                return {
                    "error": None,
                    "results": [result],
                    "summary": summary
                }
            else:
                return {
                    "error": f"Server error: {result['error']}",
                    "results": [result],
                    "summary": {"mode": "Single Server", "status": "❌ Failed"}
                }
                
        elif query_mode == "Single Server (Gojek)":
            # Query second server (Gojek)
            result = await client.query_single_server(user_query, client.server_urls[1], 1)
            
            if result["success"]:
                summary = {
                    "mode": "Single Server",
                    "server": "Gojek (7862)",
                    "status": "✅ Success",
                    "conversation_id": result.get('conversation_id', 'N/A'),
                }
                return {
                    "error": None,
                    "results": [result],
                    "summary": summary
                }
            else:
                return {
                    "error": f"Server error: {result['error']}",
                    "results": [result],
                    "summary": {"mode": "Single Server", "status": "❌ Failed"}
                }
                
        elif query_mode == "Parallel (Both Servers)":
            # Query both servers in parallel
            results = await client.query_parallel(user_query, client.server_urls)
            
            successful_results = [r for r in results if r["success"]]
            failed_results = [r for r in results if not r["success"]]
            
            summary = {
                "mode": "Parallel",
                "servers": ["Grab (7860)", "Gojek (7862)"],
                "successful": len(successful_results),
                "failed": len(failed_results),
                "total": len(results),
                "status": f"✅ {len(successful_results)}/{len(results)} successful",
            }
            
            return {
                "error": None if successful_results else "All servers failed",
                "results": results,
                "summary": summary
            }
            
    except Exception as e:
        traceback.print_exc()
        print(f"{color_red}Error occurred: {e}{color_reset}")
        return {
            "error": f"Execution error: {str(e)}",
            "results": None,
            "summary": {"status": "❌ Exception occurred"}
        }

async def gradio_query_handler(user_query: str, query_mode: str):
    """Gradio-compatible async handler for query processing."""
    try:
        # Process the query asynchronously
        result = await process_query(user_query, query_mode)
        
        # Format output for Gradio
        if result["error"]:
            status_message = f"❌ Error: {result['error']}"
            json_output = {"error": result["error"]}
        else:
            status_message = f"✅ Query processed successfully!"
            json_output = {
                "summary": result["summary"],
                "results": result["results"]
            }
        
        return status_message, json.dumps(json_output, indent=2)
        
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        error_json = {"error": str(e), "traceback": traceback.format_exc()}
        return error_msg, json.dumps(error_json, indent=2)

def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="MCP Client - Multi-Server Query Interface") as interface:
        gr.Markdown("# 🚀 MCP Client - Multi-Server Query Interface")
        gr.Markdown("Query multiple MCP servers (Grab & Gojek) and see parallel results")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(
                    label="Your Query",
                    placeholder="e.g., 'I want to go from Orchard Road to MBS' or 'Order pizza from Pizza Hut to my home'",
                    lines=3
                )
                
                with gr.Row():
                    query_mode = gr.Radio(
                        choices=["Single Server (Grab)", "Single Server (Gojek)", "Parallel (Both Servers)"],
                        label="Query Mode",
                        value="Parallel (Both Servers)"
                    )
                
                submit_btn = gr.Button("🔍 Submit Query", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📖 Instructions")
                gr.Markdown("""
                - **Single Server**: Query only Grab or Gojek
                - **Parallel**: Query both servers simultaneously
                - **Full Response**: Include LLM response + MCP calls
                - **MCP Only**: Show only tool calls and results
                
                **Example Queries:**
                - Transport: "I want to go from Orchard Road to MBS"
                - Food: "Order pizza from Pizza Hut to my home"
                """)
        
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1
            )
        
        with gr.Row():
            json_output = gr.Code(
                label="Results (JSON)",
                language="json",
                lines=20
            )
        
        # Event handler
        submit_btn.click(
            fn=gradio_query_handler,
            inputs=[user_input, query_mode],
            outputs=[status_output, json_output]
        )
        
        # Example inputs
        gr.Examples(
            examples=[
                ["I want to go from Orchard Road to Marina Bay Sands at 6pm", "Parallel (Both Servers)"],
                ["I want to go from Orchard Road to Marina Bay Sands", "Parallel (Both Servers)"],
                ["Book ride to Marina Bay Sands", "Parallel (Both Servers)"],
                    ["Order Big Mac and fries from McDonald's to my home at 123 Main Street", "Parallel (Both Servers)"],
                        ["Book a GrabTaxi from Changi Airport to CBD", "Single Server (Grab)"],
                        ["Order nasi lemak from local restaurant to office", "Single Server (Gojek)"],
            ],
            inputs=[user_input, query_mode]
        )
    
    return interface

def main():
    """Main entry point for the Gradio app."""
    print(f"{color_blue}Starting MCP Client Gradio Interface...{color_reset}")
    
    # Initialize client on startup
    success, message = initialize_client()
    print(f"{color_blue}{message}{color_reset}")
    
    # Create and launch Gradio interface
    interface = create_gradio_interface()
    interface.launch(
        server_port=7863,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()

"""
Usage:
1. Start the MCP servers:
    clear;
    if :; then
        python MCP_server_grab.py --port 7860 &
        python MCP_server_gojek.py --port 7862 &
    fi
    jobs
    wait


2. Start this Gradio app:
    clear; \
    python mistral_client_remote.py

3. Open browser to http://localhost:7863

Example queries:
    I want to go from Orchard Road to MBS
    I want to go from Woodlands to CBD
    Order pizza from Pizza Hut to my home
    Order pizza from KFC to office
"""

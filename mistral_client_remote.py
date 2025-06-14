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

# Conversation history storage
conversation_history = {
    "grab": {"messages": []},
    "gojek": {"messages": []},
    "parallel": {"messages": []}
}

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
    
    async def query_single_server(self, user_input: str, server_url: str, server_index: int = 0, conversation_messages: List[Dict] = None) -> dict:
        """Send a query to a single MCP server and get a response.
        
        Args:
            user_input: The user's query
            server_url: URL of the specific server to query
            server_index: Index of the server (for identification)
            conversation_messages: Previous conversation messages for context
            
        Returns:
            Dictionary containing the response with server information
        """
        print(f"{color_blue}\nProcessing query on server {server_index + 1} ({server_url}): {user_input}{color_reset}")
        if conversation_messages:
            print(f"{color_blue}Using conversation history with {len(conversation_messages)} messages{color_reset}")
        
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
                
                # Prepare inputs with conversation history
                print(f"{color_yellow}# Prepare inputs with conversation history{color_reset}")
                if conversation_messages:
                    # Build conversation history including the new user message
                    messages = conversation_messages + [{"role": "user", "content": user_input}]
                    inputs = messages
                    print(f"conversation_messages = {color_red}{conversation_messages}{color_reset}")
                    print(f"user_input = {color_red}{user_input}{color_reset}")
                    print(f"inputs = {color_red}{inputs}{color_reset}")
                else:
                    # First message in conversation
                    inputs = user_input
                
                # Run the query with conversation history
                print(f"{color_yellow}# Run the query{color_reset}")
                run_result = await self.client.beta.conversations.run_async(
                    run_ctx=run_ctx,
                    inputs=inputs,
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
                    "error": None,
                    "server_url": server_url,
                    "server_index": server_index,
                }

        except Exception as e:
            traceback.print_exc()
            response_output_dict = {
                "success": False,
                "response": None,
                "error": str(e),
                "server_url": server_url,
                "server_index": server_index,
            }
            print(f"{color_blue}Error occurred on server {server_index + 1}: {e}{color_reset}")

        return response_output_dict

    async def query_parallel(self, user_input: str, server_urls: List[str] = [], conversation_messages_list: List[List[Dict]] = None) -> List[dict]:
        """Send a query to multiple MCP servers in parallel and return merged results.
        
        Args:
            user_input: The user's query
            server_urls: Optional list of specific server URLs to query. 
                        If None, uses all configured servers.
            conversation_messages_list: Optional list of conversation histories for each server
            
        Returns:
            List of dictionaries containing responses from all servers
        """
        if len(server_urls) < 2:
            raise ValueError("At least 2 servers are required for parallel queries")
        
        print(f"{color_blue}\n{'='*60}{color_reset}")
        print(f"{color_blue}Starting parallel queries to {len(server_urls)} servers{color_reset}")
        print(f"{color_blue}Query: {user_input}{color_reset}")
        if conversation_messages_list:
            print(f"{color_blue}Using conversation histories: {[len(msgs) if msgs else 0 for msgs in conversation_messages_list]}{color_reset}")
        print(f"{color_blue}{'='*60}{color_reset}")
        
        # Create tasks for parallel execution
        tasks = []
        for i, server_url in enumerate(server_urls):
            conv_messages = conversation_messages_list[i] if conversation_messages_list and i < len(conversation_messages_list) else None
            task = self.query_single_server(user_input, server_url, i, conv_messages)
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

def clear_conversation_history():
    """Clear all conversation history."""
    global conversation_history
    conversation_history = {
        "grab": {"messages": []},
        "gojek": {"messages": []},
        "parallel": {"messages": []}
    }
    print(f"{color_blue}Conversation history cleared{color_reset}")
    return "🗑️ Conversation history cleared!"

def get_conversation_summary():
    """Get a summary of current conversation history."""
    global conversation_history
    summary = {}
    for mode, data in conversation_history.items():
        summary[mode] = {
            "message_count": len(data["messages"]),
            "past_messages": data["messages"] if data["messages"] else None
        }
    return json.dumps(summary, indent=2, ensure_ascii=False)

async def process_query(user_query: str, query_mode: str, maintain_history: bool = True):
    """Process a user query through the MCP client with conversation history support.
    
    Args:
        user_query: The user's query
        query_mode: Mode of querying (single server or parallel)
        maintain_history: Whether to maintain conversation history
    """
    global client, conversation_history
    
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
            
            # Get conversation messages if maintaining history
            conv_messages = conversation_history["grab"]["messages"] if maintain_history else None
            
            result = await client.query_single_server(user_query, client.server_urls[0], 0, conv_messages)
            """ result = {
                    'error': None,
                    'response': {'result': 'Your ride has been booked. The driver will arrive '
                                            'shortly.',
                                'result_dict': {'Destination': 'Marina Bay Sands',
                                                'Pickup Point': '<Current GPS location>',
                                                'Scheduled': False,
                                                'Timing': '2025-06-14T22:56:05',
                                                'Transport Type': 'JustGrab',
                                                'platform': 'Grab'},
                                'status': True},
                    'server_index': 0,
                    'server_url': 'http://127.0.0.1:7860/gradio_api/mcp/sse',
                    'success': True,
                }
            """
            
            # Update conversation history
            if maintain_history and result["success"]:
                # Add user message
                conversation_history["grab"]["messages"].append({"role": "user", "content": user_query})
                # Add assistant response
                assistant_response = result["response"].get("result", "") if result["response"] else ""
                assistant_response += json.dumps(result["response"].get("result_dict", {}), indent=4, ensure_ascii=False) if result["response"] else ""
                conversation_history["grab"]["messages"].append({"role": "assistant", "content": assistant_response})
            
            if result["success"]:
                summary = {
                    "mode": "Single Server",
                    "server": "Grab (7860)",
                    "status": "✅ Success",
                    "history_maintained": maintain_history,
                    "message_count": len(conversation_history["grab"]["messages"]) if maintain_history else 0,
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
            conv_messages = conversation_history["gojek"]["messages"] if maintain_history else None
            
            result = await client.query_single_server(user_query, client.server_urls[1], 1, conv_messages)
            
            # Update conversation history
            if maintain_history and result["success"]:
                # Add user message
                conversation_history["gojek"]["messages"].append({"role": "user", "content": user_query})
                # Add assistant response
                assistant_response = result["response"].get("result", "") if result["response"] else ""
                assistant_response += json.dumps(result["response"].get("result_dict", {}), indent=4, ensure_ascii=False) if result["response"] else ""
                conversation_history["gojek"]["messages"].append({"role": "assistant", "content": assistant_response})
            
            if result["success"]:
                summary = {
                    "mode": "Single Server",
                    "server": "Gojek (7862)",
                    "status": "✅ Success",
                    "history_maintained": maintain_history,
                    "message_count": len(conversation_history["gojek"]["messages"]) if maintain_history else 0,
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
            conv_messages_list = None
            if maintain_history:
                conv_messages_list = [
                    conversation_history["grab"]["messages"],
                    conversation_history["gojek"]["messages"]
                ]
            
            results = await client.query_parallel(user_query, client.server_urls, conv_messages_list)
            
            # Update conversation history for successful results
            if maintain_history:
                for result in results:
                    if result["success"]:
                        assistant_response = result["response"].get("result", "") if result["response"] else ""
                        assistant_response += json.dumps(result["response"].get("result_dict", {}), indent=4, ensure_ascii=False) if result["response"] else ""
                        if result["server_index"] == 0:  # Grab
                            if not conversation_history["grab"]["messages"] or conversation_history["grab"]["messages"][-1]["content"] != user_query:
                                conversation_history["grab"]["messages"].append({"role": "user", "content": user_query})
                            conversation_history["grab"]["messages"].append({"role": "assistant", "content": assistant_response})
                        elif result["server_index"] == 1:  # Gojek
                            if not conversation_history["gojek"]["messages"] or conversation_history["gojek"]["messages"][-1]["content"] != user_query:
                                conversation_history["gojek"]["messages"].append({"role": "user", "content": user_query})
                            conversation_history["gojek"]["messages"].append({"role": "assistant", "content": assistant_response})
            
            successful_results = [r for r in results if r["success"]]
            failed_results = [r for r in results if not r["success"]]
            
            summary = {
                "mode": "Parallel",
                "servers": ["Grab (7860)", "Gojek (7862)"],
                "successful": len(successful_results),
                "failed": len(failed_results),
                "total": len(results),
                "status": f"✅ {len(successful_results)}/{len(results)} successful",
                "history_maintained": maintain_history,
                "message_counts": {
                    "grab": len(conversation_history["grab"]["messages"]) if maintain_history else 0,
                    "gojek": len(conversation_history["gojek"]["messages"]) if maintain_history else 0,
                }
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

async def gradio_query_handler(user_query: str, query_mode: str, maintain_history: bool):
    """Gradio-compatible async handler for query processing."""
    try:
        # Process the query asynchronously
        result = await process_query(user_query, query_mode, maintain_history)
        
        # Prepare default containers
        grab_dict: Dict[str, Any] = {}
        gojek_dict: Dict[str, Any] = {}
        # NEW: containers for plain-text results
        grab_result_str: str = ""
        gojek_result_str: str = ""
        
        # Populate the per-platform dicts if we have any results
        if result.get("results"):
            for res in result["results"]:
                print(f"{color_yellow}res = {json.dumps(res, indent=4, ensure_ascii=False)}{color_reset}")
                # Determine which platform this result belongs to
                platform_is_grab = res.get("server_index") == 0 or str(res.get("server_url", "")).endswith(":7860/gradio_api/mcp/sse")
                platform_is_gojek = res.get("server_index") == 1 or str(res.get("server_url", "")).endswith(":7862/gradio_api/mcp/sse")

                # Extract the structured result_dict if the call succeeded
                if res.get("success") and res.get("response"):
                    # NEW: extract plain text result string
                    extracted_result = res["response"].get("result", "")
                    extracted_dict = res["response"].get("result_dict", {})
                print(f"{color_yellow}extracted_result = {extracted_result}{color_reset}")
                print(f"{color_yellow}extracted_dict = {extracted_dict}{color_reset}")

                if platform_is_grab:
                    grab_dict = extracted_dict
                    grab_result_str = extracted_result
                elif platform_is_gojek:
                    gojek_dict = extracted_dict
                    gojek_result_str = extracted_result

        # Build messages / JSON payloads for the three code components
        if result["error"]:
            status_message = f"❌ Error: {result['error']}"
            overall_json_obj = {"error": result["error"]}
        else:
            status_message = "✅ Query processed successfully!"
            overall_json_obj = {
                "summary": result["summary"],
                "results": result["results"],
            }
        
        # Get conversation history summary
        history_summary = get_conversation_summary()
        
        # Return strings for each Code component (Grab, Gojek, Overall, History)
        return (
            status_message,
            grab_result_str,
            gojek_result_str,
            json.dumps(grab_dict, indent=4, ensure_ascii=False),
            json.dumps(gojek_dict, indent=4, ensure_ascii=False),
            json.dumps(overall_json_obj, indent=4, ensure_ascii=False),
            history_summary,
        )
        
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        error_json = {"error": str(e), "traceback": traceback.format_exc()}
        # On unexpected exception, still return placeholders for all boxes
        return error_msg, \
            "", \
            "", \
            json.dumps({}), \
            json.dumps({}), \
            json.dumps(error_json, indent=4, ensure_ascii=False), \
            get_conversation_summary()

def create_gradio_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="MCP Client - Multi-Server Query Interface") as interface:
        gr.Markdown("# 🚀 MCP Client - Multi-Server Query Interface")
        gr.Markdown("Query multiple MCP servers (Grab & Gojek) with conversation history support")
        
        with gr.Row():
            with gr.Column(scale=1):
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
                
                with gr.Row():
                    maintain_history = gr.Checkbox(
                        label="Maintain Conversation History",
                        value=True,
                        info="Keep conversation context for follow-up questions"
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("🔍 Submit Query", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                
            with gr.Column(scale=1):
                # Example inputs
                gr.Examples(
                    examples=[
                        ["I want to go from Orchard Road to Marina Bay Sands at 6pm", "Parallel (Both Servers)", True],
                        ["I want to go from Orchard Road to Marina Bay Sands", "Parallel (Both Servers)", True],
                        ["Book ride to Marina Bay Sands", "Parallel (Both Servers)", True],
                        ["Order Big Mac and fries from McDonald's to my home at 123 Main Street", "Parallel (Both Servers)", True],
                        ["What's the cheapest option?", "Parallel (Both Servers)", True],  # Follow-up question
                        ["Can you make it faster?", "Parallel (Both Servers)", True],  # Follow-up question
                    ],
                    inputs=[user_input, query_mode, maintain_history]
                )
        
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1
            )

        # --- individual platform JSON outputs ---
        with gr.Row():
            with gr.Column(scale=1):
                grab_text_output = gr.Textbox(
                    label="Grab Message",
                    interactive=False,
                    lines=4,
                )
                grab_json_output = gr.Code(
                    label="Grab Response (JSON)",
                    language="json",
                    lines=12,
                )
            
            with gr.Column(scale=1):
                gojek_text_output = gr.Textbox(
                    label="Gojek Message",
                    interactive=False,
                    lines=4,
                )
                gojek_json_output = gr.Code(
                    label="Gojek Response (JSON)",
                    language="json",
                    lines=12,
                )

        # Existing aggregated results box
        with gr.Row():
            json_output = gr.Code(
                label="Results (JSON)",
                language="json",
                lines=20,
            )
        
        # Conversation history display
        with gr.Row():
            history_output = gr.Code(
                label="Conversation History Summary",
                language="json",
                lines=10,
            )
        
        # Event handlers
        submit_btn.click(
            fn=gradio_query_handler,
            inputs=[user_input, query_mode, maintain_history],
            outputs=[
                status_output,
                grab_text_output, gojek_text_output,
                grab_json_output, gojek_json_output,
                json_output,
                history_output,
            ],
        )
        
        clear_btn.click(
            fn=clear_conversation_history,
            inputs=[],
            outputs=[status_output],
        ).then(
            fn=lambda: get_conversation_summary(),
            inputs=[],
            outputs=[history_output],
        )
        
        # Instructions
        gr.Markdown("### 📖 Instructions")
        gr.Markdown("""
        - **Single Server**: Query only Grab or Gojek
        - **Parallel**: Query both servers simultaneously
        - **Conversation History**: Enable to maintain context for follow-up questions
        - **Clear History**: Reset all conversation contexts
        
        **Example Queries:**
        - Transport: "I want to go from Orchard Road to MBS"
        - Food: "Order pizza from Pizza Hut to my home"
        - Follow-up: "What's the cheapest option?" (after initial query)
        - Follow-up: "Can you make it faster?" (after booking)
        """)
    
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
        mcp_server=True,
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

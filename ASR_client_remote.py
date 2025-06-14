#!/usr/bin/env python
"""
ASR_client_remote.py - Speech-to-text front-end that streams recognised text to the
multi-server aggregator exposed by `mistral_client_remote.py` (now running as an MCP
server on http://127.0.0.1:7863).

Workflow:
1. Record or upload an audio clip (microphone/upload via Gradio UI).
2. Transcribe the audio to text.
3. Send the recognised text - together with a user-selectable query mode - to the
   MCP aggregator via the Gradio Client SDK.
4. Display both the recognised text and the aggregator JSON response.

To start the application:
    python ASR_client_remote.py
Point your browser to http://localhost:7864

The script assumes that:
    * `mistral_client_remote.py` is already running on port 7863 with
      `mcp_server=True`.
"""

import os
import pdb
import json
import traceback
from typing import Tuple, Any, Dict
from pathlib import Path
import glob

import gradio as gr
import soundfile as sf
from gradio_client import Client

COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"

# -----------------------------------------------------------------------------
# Configuration - adjust if your aggregator is not on localhost:7863
# -----------------------------------------------------------------------------
# Instantiate a persistent Gradio client once at startup
AGGREGATOR_URL = "http://127.0.0.1:7863"
print(f"{COLOR_CYAN}Connecting to aggregator at {AGGREGATOR_URL}…{COLOR_RESET}")
aggregator_client = Client(AGGREGATOR_URL)
print(f"{COLOR_CYAN}Connection established!{COLOR_RESET}")

# Debug: Show available endpoints
try:
    print(f"{COLOR_CYAN}Available endpoints: {aggregator_client.endpoints}{COLOR_RESET}")
    print(f"{COLOR_CYAN}API info: {aggregator_client.view_api()}{COLOR_RESET}")
except Exception as e:
    print(f"{COLOR_CYAN}Could not get endpoint info: {e}{COLOR_RESET}")

# -----------------------------------------------------------------------------
# ASR utilities
# -----------------------------------------------------------------------------
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
# Create OpenAI client
load_dotenv(find_dotenv())  # read local .env file
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set in environment"
OPENAI_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(audio_path: str) -> str:
    """
    https://platform.openai.com/docs/guides/speech-to-text
    model = str
        "whisper-1"
        "gpt-4o-mini-transcribe"
        "gpt-4o-transcribe"
    """
    try:
        print(f"{COLOR_CYAN}audio_path = {audio_path}{COLOR_RESET}")
        # ASR
        audio_file = open(audio_path, "rb")
        transcription = OPENAI_client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file
        )

        # Extract transcription text
        transcription_json = transcription.json()
        transcription_text = transcription.text
        print(f"{COLOR_CYAN}Transcription: {transcription_text}{COLOR_RESET}")
        """
            transcription.json = '{"text":"Testing 1 2 3","logprobs":null}'
            transcription.text = "Testing 1 2 3"
        """
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        raise e

    return transcription_text.strip()

# -----------------------------------------------------------------------------
# Audio file utilities
# -----------------------------------------------------------------------------
def load_default_audio_files():
    """Load default audio files from the audio_files directory."""
    audio_dir = Path("audio_files")
    if not audio_dir.exists():
        print(f"{COLOR_CYAN}Audio directory not found: {audio_dir}{COLOR_RESET}")
        return []
    
    # Supported audio formats
    audio_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(str(audio_dir / ext)))
    
    # Sort files for consistent ordering
    audio_files.sort()
    
    print(f"{COLOR_CYAN}Found {len(audio_files)} default audio files:{COLOR_RESET}")
    for file in audio_files:
        print(f"{COLOR_CYAN}  - {file}{COLOR_RESET}")
    
    return audio_files

# Load default audio files at startup
DEFAULT_AUDIO_FILES = load_default_audio_files()

# -----------------------------------------------------------------------------
# Main handler that ties everything together
# -----------------------------------------------------------------------------
def clear_history() -> str:
    """Clear conversation history on the aggregator server.
    
    Returns:
        Status message indicating success or failure
    """
    global aggregator_client
    try:
        # Call the clear history endpoint
        response = aggregator_client.predict(api_name="/clear_conversation_history")
        print(f"{COLOR_CYAN}Clear history response: {response}{COLOR_RESET}")
        return response if isinstance(response, str) else "🗑️ Conversation history cleared!"
    except Exception as e:
        traceback.print_exc()
        print(f"{COLOR_CYAN}Clear history failed: {e}{COLOR_RESET}")
        return f"❌ Error: Failed to clear history: {e}"

def transcribe_and_query(audio: str, query_mode: str, maintain_history: bool = True) -> Tuple[str, str, str, str, str, str, str, str]:
    """Gradio click handler: audio → text → aggregator → structured outputs.

    Returns (in order):
        recognised_text,
        status_message,
        grab_reasoning_str,
        gojek_reasoning_str,
        grab_json_str,
        gojek_json_str,
        overall_json_str,
        history_summary_str,
    """
    if not audio:
        return (
            "",                           # recognised_text
            "❌ Error: No audio supplied",  # status_message
            "", "",                     # grab_reasoning_str, gojek_reasoning_str
            json.dumps({}, indent=4),     # grab_json_str
            json.dumps({}, indent=4),     # gojek_json_str
            json.dumps({"error": "No audio supplied"}, indent=4),  # overall_json_str
            json.dumps({}, indent=4),     # history_summary_str
        )

    # 1. Transcribe audio to text
    try:
        # Perform ASR here
        # recognised_text = "Book taxi from Woodlands to MBS at 6pm"
        recognised_text = transcribe_audio(audio)
    except Exception as e:
        traceback.print_exc()
        return (
            "",                           # recognised_text
            f"❌ Error: ASR failed: {e}",  # status_message
            "", "",                     # grab_reasoning_str, gojek_reasoning_str
            json.dumps({"error": f"ASR failed: {e}"}, indent=4),     # grab_json_str
            json.dumps({"error": f"ASR failed: {e}"}, indent=4),     # gojek_json_str
            json.dumps({"error": f"ASR failed: {e}"}, indent=4),  # overall_json_str
            json.dumps({"error": f"ASR failed: {e}"}, indent=4),  # history_summary_str
        )

    if not recognised_text:
        return (
            "",                           # recognised_text
            "❌ Error: Could not recognise speech",  # status_message
            "", "",                     # grab_reasoning_str, gojek_reasoning_str
            json.dumps({"error": "Could not recognise speech"}, indent=4),     # grab_json_str
            json.dumps({"error": "Could not recognise speech"}, indent=4),     # gojek_json_str
            json.dumps({"error": "Could not recognise speech"}, indent=4),  # overall_json_str
            json.dumps({"error": "Could not recognise speech"}, indent=4),  # history_summary_str
        )

    # 2. Forward recognised text to the aggregator
    global aggregator_client  # may be None if initial connection failed
    try:
        # Try to call the main submit endpoint - this should correspond to the submit button
        response = aggregator_client.predict(recognised_text, query_mode, maintain_history, api_name="/gradio_query_handler")
    except Exception as e:
        traceback.print_exc()
        print(f"{COLOR_CYAN}[gradio_query_handler] failed: {e}{COLOR_RESET}")
        # Return error response
        return (
            recognised_text,
            f"❌ Error: Failed to call aggregator: {e}",
            "", "",
            "",
            "",
            "",
            "",
        )

    # If the aggregator returned the expected 7-element tuple, unpack it.
    status_message = ""
    grab_reasoning_str = ""
    gojek_reasoning_str = ""
    grab_json_str = json.dumps({}, indent=4)
    gojek_json_str = json.dumps({}, indent=4)
    overall_json_str = json.dumps({}, indent=4)
    history_summary_str = json.dumps({}, indent=4)

    if isinstance(response, (list, tuple)) and len(response) == 7:
        (
            status_message,
            grab_reasoning_str,
            gojek_reasoning_str,
            grab_json_str,
            gojek_json_str,
            overall_json_str,
            history_summary_str,
        ) = response
        # Ensure code outputs are strings (they might already be)
        grab_json_str = (
            grab_json_str if isinstance(grab_json_str, str) else json.dumps(grab_json_str, indent=4, ensure_ascii=False)
        )
        gojek_json_str = (
            gojek_json_str if isinstance(gojek_json_str, str) else json.dumps(gojek_json_str, indent=4, ensure_ascii=False)
        )
        overall_json_str = (
            overall_json_str if isinstance(overall_json_str, str) else json.dumps(overall_json_str, indent=4, ensure_ascii=False)
        )
        history_summary_str = (
            history_summary_str if isinstance(history_summary_str, str) else json.dumps(history_summary_str, indent=4, ensure_ascii=False)
        )
    else:
        # Unexpected shape – treat whole response as overall JSON
        status_message = f"⚠️ Aggregator returned unexpected format (expected 7 elements, got {len(response) if isinstance(response, (list, tuple)) else 'non-tuple'})"
        overall_json_str = (
            json.dumps(response, indent=4, ensure_ascii=False)
            if not isinstance(response, str)
            else response
        )

    return (
        recognised_text,
        status_message,
        grab_reasoning_str,
        gojek_reasoning_str,
        grab_json_str,
        gojek_json_str,
        overall_json_str,
        history_summary_str,
    )

# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="ASR → MCP Aggregator") as demo:
    gr.Markdown("# 🎙️ ASR Client → MCP Aggregator")
    gr.Markdown("Record or upload speech, then pass the recognised text to the **Mistral MCP** aggregator.")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Input Audio")
        with gr.Column():
            query_mode = gr.Radio(
                ["Single Server (Grab)", "Single Server (Gojek)", "Parallel (Both Servers)"],
                value="Parallel (Both Servers)",
                label="Query Mode",
            )
            maintain_history = gr.Checkbox(
                label="Maintain Conversation History",
                value=True,
                info="Keep conversation context for follow-up questions"
            )

    # Add audio examples section
    if DEFAULT_AUDIO_FILES:
        with gr.Row():
            gr.Markdown("### 🎵 Default Audio Examples")
            gr.Examples(
                examples=[[audio_file] for audio_file in DEFAULT_AUDIO_FILES],
                inputs=[audio_input],
                label="Click to load sample audio files"
            )

    with gr.Row():
        submit_btn = gr.Button("✨ Transcribe & Query", variant="primary")
        clear_btn = gr.Button("🗑️ Clear History", variant="secondary")

    # --- ASR outputs ---
    with gr.Row():
        with gr.Column(scale=1):
            recognised_text_output = gr.Textbox(label="Recognised Text", interactive=False)
        with gr.Column(scale=1):
            status_message_output = gr.Textbox(label="Status Message", interactive=False)
    
    # --- individual platform outputs with reasoning ---
    with gr.Row():
        with gr.Column(scale=1):
            grab_reasoning_output = gr.Textbox(label="Grab Reasoning", interactive=False, lines=6)
            grab_json_output = gr.Code(label="Grab Response (JSON)", language="json")
        with gr.Column(scale=1):
            gojek_reasoning_output = gr.Textbox(label="Gojek Reasoning", interactive=False, lines=6)
            gojek_json_output = gr.Code(label="Gojek Response (JSON)", language="json")
    
    overall_json_output = gr.Code(label="Overall Response (JSON)", language="json")
    
    # Conversation history display
    with gr.Row():
        history_output = gr.Code(
            label="Conversation History Summary",
            language="json",
            lines=10,
        )

    submit_btn.click(
        fn=transcribe_and_query,
        inputs=[audio_input, query_mode, maintain_history],
        outputs=[
            recognised_text_output, status_message_output,
            grab_reasoning_output, gojek_reasoning_output,
            grab_json_output, gojek_json_output,
            overall_json_output, history_output
        ],
    )

    clear_btn.click(
        fn=clear_history,
        outputs=[status_message_output],
    )

    # Add instructions section
    gr.Markdown("### 📖 Instructions")
    gr.Markdown(f"""
    - **Record Audio**: Use the microphone button to record your voice
    - **Upload Audio**: Click the upload button to select an audio file
    - **Default Examples**: Use the sample audio files above to test the system
    - **Query Modes**: Choose between single server or parallel processing
    - **History**: Enable to maintain conversation context
    - **Reasoning**: Shows the AI's reasoning process for each platform's response
    
    **Available Sample Files**: {len(DEFAULT_AUDIO_FILES)} audio files loaded from `audio_files/` directory
    """)

if __name__ == "__main__":
    demo.launch(server_port=7864, share=False, debug=True)
    
"""
cd ~/Codes/LLM/OpenAI/gradio_mcp

conda deactivate
conda activate gradio_mcp

clear; \
    python ASR_client_remote.py

"""

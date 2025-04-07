import gradio as gr
import soundfile as sf
import pdb
import json

from util import NC, GREEN
from grab_service_router import GrabServiceRouter

# ----------------------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------------------
def perform_ASR(audio):
    """
    https://platform.openai.com/docs/guides/speech-to-text
    model = str
        "whisper-1"
        "gpt-4o-mini-transcribe"
        "gpt-4o-transcribe"
    """
    try:
        # Get recording data
        sr, data = audio

        # Save audio data to mp3 file
        wav_path = "./audio/tmp/audio.wav"
        sf.write(wav_path, data, sr)

        # ASR
        audio_file = open(wav_path, "rb")
        transcription = grab_service_router.client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file
        )

        # Extract transcription text
        transcription_json = transcription.json()
        transcription_text = transcription.text
        """
            transcription.json = '{"text":"Testing 1 2 3","logprobs":null}'
            transcription.text = "Testing 1 2 3"
        """
    except Exception as e:
        transcription_text = f"Error: {e}"
        print(transcription_text)
        pdb.set_trace()

    return transcription_text

# ----------------------------------------------------------------------------------------
# GrabServiceRouter functions
# ----------------------------------------------------------------------------------------
grab_service_router = GrabServiceRouter()

def get_relevant_image(service_router_response):
    # Get the image path
    
    # Hack
    if service_router_response['service'] == "null":
        image_path = "./images/grab/HomePage.jpg"
    elif service_router_response['service'] == "GrabTransport":
        image_path = "./images/grab/GrabTransport_bookride.jpg"
    elif service_router_response['service'] == "GrabFood":
        image_path = "./images/grab/GrabFood_filter.jpg"
    elif service_router_response['service'] == "GrabPay":
        image_path = "./images/grab/scanner.jpg"
    else:
        image_path = "./images/grab/HomePage.jpg"
    return image_path

def get_response_text(user_input):
    # Process the user input and return a response
    service_router_response = grab_service_router.run(user_input)
    service_router_response_str = json.dumps(service_router_response, indent=4)
    print(f"service_router_response_str: {GREEN}{service_router_response_str}{NC}")
    # Get the image path
    image_path = get_relevant_image(service_router_response)
    print(f"image_path: {GREEN}{image_path}{NC}")
    return service_router_response_str, image_path

def get_response_audio(audio):
    # perform ASR
    user_input = perform_ASR(audio)
    print(f"user_input: {GREEN}{user_input}{NC}")
    # perform LLM response
    service_router_response_str, image_path = get_response_text(user_input)
    return user_input, service_router_response_str, image_path

# ----------------------------------------------------------------------------------------
# Debug
# ----------------------------------------------------------------------------------------
def debugger_here():
    pdb.set_trace()
    return 

# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    # Display the image with existing boxes for annotation
    with gr.Blocks() as app:
        gr.HTML("""
        <h1 style='text-align: center'>
        Omni Wombat Agent
        </h1>
        """)
        with gr.Row():
            with gr.Column():
                # User voice
                user_voice = gr.Audio(
                    sources=["microphone"],
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#01C6FF",
                        waveform_progress_color="#0066B4",
                        skip_length=2,
                        show_controls=False,
                    ),
                )
                # User query
                user_query_textbox = gr.Textbox(
                    label="User Query (Text)",
                    scale=0.8,
                )

                # Add template input texts
                template_v1_1 = "Book grabhitch from NUS LT27 to UOB Plaza at 6 fifteen pm"; template_v1_1_wavpath = "./audio/grab/v1-book_GrabHitch_at_615pm.wav"
                template_v1_2 = "Book grabshare from home to gym"; template_v1_2_wavpath = "./audio/grab/v1-book_GrabShare_Home2Gym.wav"
                template_v2 = "What are the halal food near me ?"; template_v2_wavpath = "./audio/grab/v2-what_halal_foods_near_me.wav"
                template_v3 = "Pay merchant QR code."; template_v3_wavpath = "./audio/grab/v3-open_QR_code_to_pay.wav"
                template_v4 = "Hi, how are you ?"; template_v4_wavpath = "./audio/grab/v4-Hello.wav"
                gr.Button(f"[Transport]: {template_v1_1}", value=template_v1_1).click(fn=lambda: (template_v1_1, template_v1_1_wavpath), outputs = [user_query_textbox, user_voice])
                gr.Button(f"[Transport]: {template_v1_2}", value=template_v1_2).click(fn=lambda: (template_v1_2, template_v1_2_wavpath), outputs = [user_query_textbox, user_voice])
                gr.Button(f"[Food]: {template_v2}", value=template_v2).click(fn=lambda: (template_v2, template_v2_wavpath), outputs = [user_query_textbox, user_voice])
                gr.Button(f"[GrabPay]: {template_v3}", value=template_v3).click(fn=lambda: (template_v3, template_v3_wavpath), outputs = [user_query_textbox, user_voice])
                gr.Button(f"[Reject]: {template_v4}", value=template_v4).click(fn=lambda: (template_v4, template_v4_wavpath), outputs = [user_query_textbox, user_voice])
                # Service router response
                output_textbox = gr.Textbox(
                    label="Service Router Response",
                    lines=10,
                    value="",
                )
            with gr.Column():
                # Display Image
                image_landing_page = gr.Image(
                    label="Image",
                    value="./images/grab/HomePage.jpg",
                    scale=0.2,
                    # height=500,
                )
        with gr.Row():
            # Action buttons
            get_response_button = gr.Button("Get Response")

        # # Debugger
        # with gr.Row():
        #     debug_button = gr.Button("Debug")

        # Start recording
        stream = user_voice.start_recording(
            inputs=[user_voice],
        )
        respond = user_voice.stop_recording(
            fn=get_response_audio,
            inputs=[user_voice],
            outputs=[user_query_textbox, output_textbox, image_landing_page]
        )

        # Text based inference
        get_response_button.click(
            fn=get_response_text,
            inputs=[user_query_textbox],
            outputs=[output_textbox, image_landing_page]
        )

        # Debugger
        # debug_button.click(
        #     fn=debugger_here,
        #     inputs=[],
        #     outputs=[]
        # )

    # Gradio Launch
    app.launch(
        server_port=5001,
        share=False,
        debug=True,
        show_error=True,
    )

if __name__ == "__main__":
    main()

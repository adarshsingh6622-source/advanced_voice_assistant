import gradio as gr
import whisper
import os
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
import uuid

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# OPENROUTER SETUP
# =========================
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# =========================
# LOAD WHISPER MODEL
# =========================
whisper_model = whisper.load_model("tiny")

# =========================
# SPEECH TO TEXT (LOCAL)
# =========================
def speech_to_text(audio_file):
    try:
        if audio_file is None:
            return ""

        result = whisper_model.transcribe(audio_file)
        return result["text"]

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# CHAT (OpenRouter)
# =========================
def generate_response(user_input):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a smart AI assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# TEXT TO SPEECH
# =========================
def text_to_speech(text):
    try:
        filename = f"voice_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        return None

# =========================
# MAIN CHAT FUNCTION
# =========================
def chat(user_input):
    if not user_input:
        return "", None

    response = generate_response(user_input)
    audio = text_to_speech(response)

    return response, audio

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Advanced AI Voice Assistant (Whisper + OpenRouter)")

    text_input = gr.Textbox(
        label="Your Message",
        placeholder="Type or speak..."
    )

    mic_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎤 Speak"
    )

    send_btn = gr.Button("Send")

    output_text = gr.Textbox(label="Assistant Response")
    output_audio = gr.Audio(label="Voice Output", autoplay=True)

    # 🎤 Mic → Text
    mic_input.stop_recording(
        fn=speech_to_text,
        inputs=mic_input,
        outputs=text_input
    )

    # 💬 Send → Chat
    send_btn.click(
        fn=chat,
        inputs=text_input,
        outputs=[output_text, output_audio]
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    demo.launch()

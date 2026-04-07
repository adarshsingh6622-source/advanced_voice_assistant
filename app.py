import streamlit as st
from utils.speech_to_text import recognize_speech
from utils.llm_brain import generate_response
from utils.text_to_speech import speak
from utils.text_to_speech import speak, stop_speaking
st.title("Advanced AI Voice Assistant")

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# Voice input
if st.button(" Start Talking"):
    text = recognize_speech()
    st.session_state.voice_text = text

# Text box
user_input = st.text_input("Your message", st.session_state.voice_text)

# Send button
if st.button("Send"):
    if user_input:

        
        stop_speaking()

        st.write("You said:", user_input)

        response = generate_response(user_input)

        st.write("Assistant:", response)

        #  voice output
        speak(response)
    
    if st.button("Stop Voice"):
        stop_speaking()
           
        
        
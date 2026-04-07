import whisper
import speech_recognition as sr

model = whisper.load_model("base")

def recognize_speech():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Listening...")

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

    with open("temp.wav", "wb") as f:
        f.write(audio.get_wav_data())

    result = model.transcribe("temp.wav")

    return result["text"]

       
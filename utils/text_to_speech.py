from gtts import gTTS
import pygame
import os
import uuid
import time

# global state
is_speaking = False

def stop_speaking():
    global is_speaking
    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except:
        pass
    is_speaking = False


def speak(text):
    global is_speaking

    if not text:
        return

    # 🔥 पहले पुरानी आवाज बंद
    stop_speaking()

    is_speaking = True

    # 🔥 हर बार unique file (IMPORTANT)
    filename = f"voice_{uuid.uuid4().hex}.mp3"

    try:
        # text to speech
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        # play sound
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # wait until finished
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.stop()
        pygame.mixer.quit()

    except Exception as e:
        print("TTS Error:", e)

    finally:
        # 🔥 safe delete (no error now)
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except:
            pass

        is_speaking = False
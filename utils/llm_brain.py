from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API Key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(" OPENROUTER_API_KEY not found in .env file")

# Client setup
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# =========================
# MAIN FUNCTION
# =========================
def generate_response(user_input):
    user_input = user_input.lower().strip()

    #  Date feature
    if "date" in user_input:
        return datetime.now().strftime("%d-%m-%Y")

    #  Time feature
    if "time" in user_input:
        return datetime.now().strftime("%H:%M:%S")

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",  # best auto model
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart AI voice assistant. Answer clearly and shortly."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "advanced-voice-assistant"
            },
            temperature=0.7,
            max_tokens=300
        )

        answer = response.choices[0].message.content

        # fallback safety
        if not answer or answer.strip() == "":
            return "Sorry, I didn't understand that."

        return answer

    except Exception as e:
        return f"Error: {str(e)}"


# =========================
# RUN LOOP (CLI TEST)
# =========================
if __name__ == "__main__":
    print(" AI Assistant started (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print(" Bye!")
            break

        response = generate_response(user_input)
        print("AI:", response)
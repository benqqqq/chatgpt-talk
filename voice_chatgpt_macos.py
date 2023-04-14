import io
import os
import wave

import requests
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
import subprocess

load_dotenv()


def chat_gpt(prompt):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("API key not found. Make sure it is set in the .env file.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    data = {
        "model": "gpt-3.5-turbo-0301",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 100,
        "n": 1,
        "stop": None,
        "temperature": 0.7,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        try:
            error_message = response.json().get("error", {}).get("message", "")
        except ValueError:
            error_message = ""

        raise Exception(
            f"Request failed with status code {response.status_code}"
            f"{'; ' + error_message if error_message else ''}"
        )

def record_audio(duration):
    samplerate = 16000
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    print("Listening...")
    myrecording = sd.rec(int(duration * samplerate))
    sd.wait()
    return myrecording


def recognize_speech():
    duration = 5  # Duration to record in seconds
    audio_data = record_audio(duration)
    audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)

    recognizer = sr.Recognizer()
    with io.BytesIO() as buffer:
        with wave.open(buffer, 'wb') as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(16000)
            wave_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        with sr.AudioFile(buffer) as source:
            audio = recognizer.record(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")


def speak_text(text):
    tts = gTTS(text, lang="en")
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        subprocess.run(["afplay", fp.name])


def main():
    while True:
        prompt = recognize_speech()
        if prompt:
            response = chat_gpt(prompt)
            print("ChatGPT says:", response)
            speak_text(response)
        else:
            print("Please try again.")


if __name__ == "__main__":
    main()

import io
import threading
import wave

import numpy as np
import sounddevice as sd
import speech_recognition as sr


def record_audio(duration):
    samplerate = 16000
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    # print("Listening...")
    myrecording = sd.rec(int(duration * samplerate))
    sd.wait()
    return myrecording


def recognize_speech(audio_data):
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
            # print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"{text}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")


def main():
    while True:
        duration = 5  # Duration to record in seconds
        audio_data = record_audio(duration)

        # Create a new thread to recognize speech in the background
        recognize_thread = threading.Thread(target=recognize_speech, args=(audio_data,))
        recognize_thread.start()


if __name__ == "__main__":
    main()

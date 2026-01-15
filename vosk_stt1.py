# vosk_stt.py
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
import os

# Load Vosk model (adjust to your extracted model path)
model_path = "C:/COLLEGE/alzheimers/models/vosk-model-small-en-in-0.4"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Vosk model not found at: {model_path}")

model = Model(model_path)
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

def recognize_speech():
    samplerate = 16000
    device = None  # Default mic
    rec = KaldiRecognizer(model, samplerate)

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                           dtype='int16', channels=1, callback=callback):
        print("🎤 Speak now (say something like 'Who is this?')...")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    print("📝 You said:", text)
                    return text

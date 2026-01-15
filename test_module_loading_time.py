import time

print("Testing individual module loading times...")

# Test vosk_stt
print("\n1. Testing vosk_stt import...")
start = time.time()
try:
    from vosk_stt import recognize_speech
    vosk_time = time.time() - start
    print(f"✅ vosk_stt loaded in {vosk_time:.1f} seconds")
except Exception as e:
    print(f"❌ vosk_stt failed: {e}")

# Test face_recognition5  
print("\n2. Testing face_recognition5 import...")
start = time.time()
try:
    from face_recognition5 import recognize_face_from_image
    face_time = time.time() - start
    print(f"✅ face_recognition5 loaded in {face_time:.1f} seconds")
except Exception as e:
    print(f"❌ face_recognition5 failed: {e}")

print(f"\n=== MODULE TIMING ===")
print(f"vosk_stt: {vosk_time:.1f} seconds")
print(f"face_recognition5: {face_time:.1f} seconds")
print(f"TOTAL: {vosk_time + face_time:.1f} seconds")
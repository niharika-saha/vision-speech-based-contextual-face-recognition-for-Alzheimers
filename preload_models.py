#!/usr/bin/env python3
"""
Pre-load all heavy models to speed up main application
Run this once before running your main controller
"""

import os
import time
import torch
import mediapipe as mp
import numpy as np
from mobilefacenet import MobileFaceNet

print("Starting model pre-loading...")
total_start = time.time()

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Pre-load MobileFaceNet
print("Loading MobileFaceNet model...")
start = time.time()
model = MobileFaceNet()
model.load_state_dict(torch.load("models/mobilefacenet.pth", map_location="cpu"))
model.eval()
print(f"   MobileFaceNet loaded in {time.time() - start:.1f} seconds")

# 2. Pre-load MediaPipe components
print("Loading MediaPipe components...")
start = time.time()
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)
mp_mesh = mp.solutions.face_mesh
print(f"   MediaPipe loaded in {time.time() - start:.1f} seconds")

# 3. Pre-load embeddings
print("Loading face embeddings...")
start = time.time()
EMBEDDING_DIR = "embeddings_multi2"
known_embeddings = {}
embedding_count = 0

for person in os.listdir(EMBEDDING_DIR):
    folder = os.path.join(EMBEDDING_DIR, person)
    if os.path.isdir(folder):
        embeddings = []
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                embeddings.append(np.load(os.path.join(folder, file)))
                embedding_count += 1
        if embeddings:
            known_embeddings[person] = embeddings

print(f"   Loaded {embedding_count} embeddings for {len(known_embeddings)} people in {time.time() - start:.1f} seconds")

# 4. Test run to warm up models
print(" Warming up models with dummy data...")
start = time.time()

# Create dummy image for testing
import cv2
import numpy as np
dummy_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

# Warm up MediaPipe
with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
    mesh.process(dummy_img)

# Warm up face detector
face_detector.process(dummy_img)

# Warm up model with dummy tensor
dummy_tensor = torch.randn(1, 3, 112, 112)
with torch.no_grad():
    model(dummy_tensor)

print(f"   Models warmed up in {time.time() - start:.1f} seconds")

total_time = time.time() - total_start
print(f"\nAll models pre-loaded successfully!")
print(f"Total pre-loading time: {total_time:.1f} seconds")
print(f"Found {len(known_embeddings)} people in database")

# Save a flag file to indicate models are ready
with open("models_ready.flag", "w") as f:
    f.write("ready")


#print("\n💡 Models are now cached in memory. Your main application should start faster!")
#print("Run your main_controller3.py now for faster startup.")"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# --- Paths ---
SOURCE_DIR = "datasets/archive"
OUTPUT_DIR = "datasets/cleaned_archive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Settings ---
MIN_FACE_SCORE = 0.8
MIN_SIZE = 80  # Minimum face size (pixels)
CROP_SIZE = 112
CROP_MARGIN = 0.1

# --- Load Face Detector ---
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# --- Detect & Crop ---
def detect_and_crop(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    results = detector.process(img_rgb)
    if not results.detections:
        return None
    det = results.detections[0]
    if det.score[0] < MIN_FACE_SCORE:
        return None

    box = det.location_data.relative_bounding_box
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    margin = int(CROP_MARGIN * max(bw, bh))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    cropped = image[y1:y2, x1:x2]
    if cropped.shape[0] < MIN_SIZE or cropped.shape[1] < MIN_SIZE:
        return None

    return cv2.resize(cropped, (CROP_SIZE, CROP_SIZE))

# --- Process All Folders ---
for person in os.listdir(SOURCE_DIR):
    person_dir = os.path.join(SOURCE_DIR, person)
    save_dir = os.path.join(OUTPUT_DIR, person)
    os.makedirs(save_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(person_dir), desc=f"Processing {person}"):
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        face = detect_and_crop(img)
        if face is not None:
            cv2.imwrite(os.path.join(save_dir, img_file), face)

print("\n✅ Done! Cleaned dataset is in:", OUTPUT_DIR)

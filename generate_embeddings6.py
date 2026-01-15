#Enhanced 5-point face alignment for better similarity scores
import os
import cv2
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.nn.functional import normalize
import mediapipe as mp
from mobilefacenet import MobileFaceNet
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- PATHS ---
DATASET_DIR = "datasets/cleaned_archive"
EMBEDDING_DIR = "embeddings_multi3" #note: updated version
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# --- LOAD MODELS ---
model = MobileFaceNet()
model.load_state_dict(torch.load("models/mobilefacenet.pth", map_location='cpu'))
model.eval()

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)
mp_mesh = mp.solutions.face_mesh

# --- ENHANCED TRANSFORM WITH BETTER NORMALIZATION ---
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

def get_5_point_landmarks(face_landmarks, w, h):
    """Extract 5 key facial landmarks for better alignment"""
    lm = face_landmarks.multi_face_landmarks[0]
    
    # Get landmark coordinates
    left_eye = np.array([lm.landmark[33].x * w, lm.landmark[33].y * h])  # Left eye corner
    right_eye = np.array([lm.landmark[263].x * w, lm.landmark[263].y * h])  # Right eye corner
    nose_tip = np.array([lm.landmark[1].x * w, lm.landmark[1].y * h])  # Nose tip
    left_mouth = np.array([lm.landmark[61].x * w, lm.landmark[61].y * h])  # Left mouth corner
    right_mouth = np.array([lm.landmark[291].x * w, lm.landmark[291].y * h])  # Right mouth corner
    
    return np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth], dtype=np.float32)

def align_face_5_point(image, landmarks):
    """Perform 5-point face alignment for better normalization"""
    h, w = image.shape[:2]
    
    # Standard 5-point template (normalized coordinates)
    template = np.array([
        [0.31556875, 0.4615741],   # Left eye
        [0.68262106, 0.4615741],   # Right eye  
        [0.5, 0.61041667],         # Nose tip
        [0.34947552, 0.84090909],  # Left mouth
        [0.65052448, 0.84090909]   # Right mouth
    ], dtype=np.float32)
    
    # Scale template to image size
    template[:, 0] *= w
    template[:, 1] *= h
    
    # Calculate similarity transform
    tform = cv2.estimateAffinePartial2D(landmarks, template)[0]
    
    if tform is not None:
        aligned = cv2.warpAffine(image, tform, (w, h))
        return aligned
    else:
        return image

# --- MAIN LOOP ---
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    save_path = os.path.join(EMBEDDING_DIR, person_name)
    os.makedirs(save_path, exist_ok=True)

    idx = 0

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Enhanced preprocessing
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=15)  # Slightly less aggressive
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- ENHANCED 5-POINT FACE ALIGNMENT ---
        with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                             refine_landmarks=True, min_detection_confidence=0.8) as mesh:
            face_landmarks = mesh.process(img_rgb)
            if not face_landmarks or not face_landmarks.multi_face_landmarks:
                continue
            
            h, w, _ = img_rgb.shape
            
            try:
                # Get 5-point landmarks
                landmarks = get_5_point_landmarks(face_landmarks, w, h)
                
                # Perform 5-point alignment
                img_rgb = align_face_5_point(img_rgb, landmarks)
                
            except Exception as e:
                print(f"Alignment failed for {img_name}: {e}")
                continue

        # --- FACE DETECTION ---
        results = face_detector.process(img_rgb)
        if not results.detections:
            continue

        detection = results.detections[0]
        if detection.score[0] < 0.8:
            continue

        bbox = detection.location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, x1 + int(bbox.width * w))
        y2 = min(h, y1 + int(bbox.height * h))

        face = img_rgb[y1:y2, x1:x2]
        if face.shape[0] < 50 or face.shape[1] < 50:
            continue

        face_pil = Image.fromarray(face)

        # --- AUGMENTATIONS ---
        augmentations = []

        # Original + common augmentations
        augmentations.append(face_pil)
        augmentations.append(TF.hflip(face_pil))
        
        # Reduced rotation angles for better quality
        augmentations.append(face_pil.rotate(5))
        augmentations.append(face_pil.rotate(-5))
        
        # Better lighting augmentations
        augmentations.append(TF.adjust_brightness(face_pil, 1.05))
        augmentations.append(TF.adjust_contrast(face_pil, 1.1))
        augmentations.append(TF.adjust_gamma(face_pil, 0.95))
        augmentations.append(TF.adjust_gamma(face_pil, 1.05))

        # --- Always Add Helmet & Sunglasses Variants ---
        helmet_path = "helmet.png"
        sunglasses_path = "sunglasses.png"
        helmet_img = Image.open(helmet_path).convert("RGBA") if os.path.exists(helmet_path) else None
        sunglasses_img = Image.open(sunglasses_path).convert("RGBA") if os.path.exists(sunglasses_path) else None

        for aug_face in augmentations:
            aug_pil = aug_face.convert("RGBA")

            # Sunglasses with better positioning
            if sunglasses_img:
                sung = aug_pil.copy()
                sg_resized = sunglasses_img.resize((int(aug_pil.width * 0.6), int(aug_pil.height * 0.2)))
                x_pos = int(aug_pil.width * 0.2)
                y_pos = int(aug_pil.height * 0.25)
                sung.paste(sg_resized, (x_pos, y_pos), sg_resized)
                sung = sung.convert("RGB")
                face_tensor = transform(sung).unsqueeze(0)
                with torch.no_grad():
                    emb = model(face_tensor)
                    emb = normalize(emb).squeeze().numpy()
                    np.save(os.path.join(save_path, f"{idx}.npy"), emb)
                    idx += 1

            # Helmet with better positioning  
            if helmet_img:
                hlm = aug_pil.copy()
                helmet_resized = helmet_img.resize((aug_pil.width, int(aug_pil.height * 0.5)))
                hlm.paste(helmet_resized, (0, 0), helmet_resized)
                hlm = hlm.convert("RGB")
                face_tensor = transform(hlm).unsqueeze(0)
                with torch.no_grad():
                    emb = model(face_tensor)
                    emb = normalize(emb).squeeze().numpy()
                    np.save(os.path.join(save_path, f"{idx}.npy"), emb)
                    idx += 1

            # Also save plain augmentation
            aug_pil_rgb = aug_pil.convert("RGB")
            face_tensor = transform(aug_pil_rgb).unsqueeze(0)
            with torch.no_grad():
                emb = model(face_tensor)
                emb = normalize(emb).squeeze().numpy()
                np.save(os.path.join(save_path, f"{idx}.npy"), emb)
                idx += 1

        print(f" Saved {idx} embeddings for {person_name}")

print("\nAll embeddings generated with enhanced 5-point face alignment.")
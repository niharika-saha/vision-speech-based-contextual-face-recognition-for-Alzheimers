#Enhanced face recognition with 5-point alignment and better similarity matching

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.nn.functional import normalize
import mediapipe as mp
from mobilefacenet import MobileFaceNet
from PIL import Image
import torchvision.transforms.functional as TF
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

EMBEDDING_DIR = "embeddings_multi3" #updated to match generation script
SIMILARITY_THRESHOLD = 0.6

# --- Load embeddings ---
known_embeddings = {}
for person in os.listdir(EMBEDDING_DIR):
    folder = os.path.join(EMBEDDING_DIR, person)
    if os.path.isdir(folder):
        embeddings = []
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                embeddings.append(np.load(os.path.join(folder, file)))
        if embeddings:
            known_embeddings[person] = embeddings

# --- Model & transforms ---
model = MobileFaceNet()
model.load_state_dict(torch.load("models/mobilefacenet.pth", map_location="cpu"))
model.eval()

# Updated transform to match generation script
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)
mp_mesh = mp.solutions.face_mesh

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
    
    # Your Dataset's Optimal Template (calculated from your actual data)
    template = np.array([
        [0.25669807, 0.36378297],   # Left eye
        [0.73767142, 0.35982665],   # Right eye
        [0.51510032, 0.61324696],   # Nose tip
        [0.35637136, 0.75871064],   # Left mouth
        [0.64822181, 0.75614430],   # Right mouth
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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def enhanced_similarity_matching(test_embedding, person_embeddings):
    """Enhanced similarity matching with multiple strategies"""
    similarities = []
    
    for emb in person_embeddings:
        sim = cosine_similarity(test_embedding, emb)
        similarities.append(sim)
    
    if not similarities:
        return 0.0
    
    # Use multiple matching strategies
    max_sim = max(similarities)
    avg_top3 = np.mean(sorted(similarities, reverse=True)[:3]) if len(similarities) >= 3 else max_sim
    
    # Weighted combination: 70% max similarity, 30% average of top 3
    final_score = 0.7 * max_sim + 0.3 * avg_top3
    
    return final_score

def preprocess_and_embed(face_pil):
    """Enhanced preprocessing with better variants"""
    variants = []

    # Base image
    variants.append(face_pil)
    
    # Horizontal flip
    variants.append(TF.hflip(face_pil))
    
    # Reduced rotation for better quality
    for angle in [-5, 5]:
        variants.append(face_pil.rotate(angle))
    
    # Better lighting adjustments
    variants.append(TF.adjust_brightness(face_pil, 1.05))
    variants.append(TF.adjust_contrast(face_pil, 1.1))
    variants.append(TF.adjust_gamma(face_pil, 0.95))
    variants.append(TF.adjust_gamma(face_pil, 1.05))

    # Overlay sunglasses with better positioning
    if os.path.exists("sunglasses.png"):
        sunglasses = Image.open("sunglasses.png").convert("RGBA")
        for base in [face_pil, TF.hflip(face_pil)]:  # Apply to original and flipped
            overlaid = base.convert("RGBA")
            sg_resized = sunglasses.resize((int(base.width * 0.6), int(base.height * 0.2)))
            x_pos = int(base.width * 0.2)
            y_pos = int(base.height * 0.25)
            overlaid.paste(sg_resized, (x_pos, y_pos), sg_resized)
            variants.append(overlaid.convert("RGB"))

    # Overlay helmet with better positioning
    if os.path.exists("helmet.png"):
        helmet = Image.open("helmet.png").convert("RGBA")
        for base in [face_pil, TF.hflip(face_pil)]:  # Apply to original and flipped
            overlaid = base.convert("RGBA")
            helmet_resized = helmet.resize((base.width, int(base.height * 0.5)))
            overlaid.paste(helmet_resized, (0, 0), helmet_resized)
            variants.append(overlaid.convert("RGB"))

    # Convert to embeddings
    embeddings = []
    for img in variants:
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = normalize(model(tensor)).squeeze().numpy()
            embeddings.append(emb)

    # Return average embedding for better stability
    return np.mean(embeddings, axis=0)

def recognize_face_from_image(image_path):
    print("Reading image...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Couldn't read image.")

    print("Converting and preprocessing...")
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=15)  # Match generation script
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    print("Enhanced 5-point face alignment...")
    start = time.time()
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                         refine_landmarks=True, min_detection_confidence=0.8) as mesh:
        result = mesh.process(img_rgb)
        end = time.time()
        print(f"Face alignment took {end - start:.2f} seconds")
        
        if not result.multi_face_landmarks:
            raise ValueError("No face landmarks detected.")
        
        try:
            # Get 5-point landmarks
            landmarks = get_5_point_landmarks(result, w, h)
            
            # Perform 5-point alignment
            img_rgb = align_face_5_point(img_rgb, landmarks)
            
        except Exception as e:
            print(f"Alignment failed, using original image: {e}")

    print("Running face detection...")
    results = face_detector.process(img_rgb)
    if not results.detections:
        raise ValueError("No face detected.")

    det = results.detections[0]
    box = det.location_data.relative_bounding_box
    x1 = max(0, int(box.xmin * w))
    y1 = max(0, int(box.ymin * h))
    x2 = min(w, x1 + int(box.width * w))
    y2 = min(h, y1 + int(box.height * h))
    face = img_rgb[y1:y2, x1:x2]

    if face.shape[0] < 50 or face.shape[1] < 50:
        raise ValueError("Face too small.")

    print("Preparing face PIL image...")
    face_pil = Image.fromarray(face)
    
    # Apply same preprocessing as in generation
    face_pil = TF.adjust_brightness(face_pil, 1.05)
    face_pil = TF.adjust_contrast(face_pil, 1.1)

    print("Embedding with enhanced preprocessing variants...")
    test_embedding = preprocess_and_embed(face_pil)

    print("Enhanced similarity matching...")
    scores = []
    for person, embs in known_embeddings.items():
        score = enhanced_similarity_matching(test_embedding, embs)
        scores.append((person, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = scores[:3]

    print("\nRecognition Top-3:")
    for i, (name, score) in enumerate(top3):
        print(f"{i+1}. {name}: {score:.4f}")

    if top3[0][1] >= SIMILARITY_THRESHOLD:
        return top3[0][0], top3[0][1]
    else:
        return None, top3[0][1]
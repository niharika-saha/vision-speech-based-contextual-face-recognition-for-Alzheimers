#Analyze your dataset to find optimal face alignment template
import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = "datasets/cleaned_archive"

# Initialize MediaPipe
mp_mesh = mp.solutions.face_mesh

def get_5_point_landmarks(face_landmarks, w, h):
    """Extract 5 key facial landmarks"""
    lm = face_landmarks.multi_face_landmarks[0]
    
    # Get landmark coordinates
    left_eye = np.array([lm.landmark[33].x, lm.landmark[33].y])  # Normalized coordinates
    right_eye = np.array([lm.landmark[263].x, lm.landmark[263].y])
    nose_tip = np.array([lm.landmark[1].x, lm.landmark[1].y])
    left_mouth = np.array([lm.landmark[61].x, lm.landmark[61].y])
    right_mouth = np.array([lm.landmark[291].x, lm.landmark[291].y])
    
    return np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth])

def analyze_dataset_landmarks():
    """Analyze your dataset to find optimal landmark positions"""
    all_landmarks = []
    processed_count = 0
    
    print("Analyzing your dataset for optimal face template...")
    
    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
            
        person_landmarks = []
        
        for img_name in os.listdir(person_path)[:5]:  # Sample 5 images per person
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            
            with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                 refine_landmarks=True, min_detection_confidence=0.8) as mesh:
                result = mesh.process(img_rgb)
                
                if result.multi_face_landmarks:
                    try:
                        landmarks = get_5_point_landmarks(result, w, h)
                        person_landmarks.append(landmarks)
                        processed_count += 1
                    except:
                        continue
        
        if person_landmarks:
            # Average landmarks for this person
            avg_landmarks = np.mean(person_landmarks, axis=0)
            all_landmarks.append(avg_landmarks)
    
    if not all_landmarks:
        print("No landmarks found in dataset!")
        return None
    
    # Calculate dataset-wide average landmark positions
    dataset_template = np.mean(all_landmarks, axis=0)
    dataset_std = np.std(all_landmarks, axis=0)
    
    print(f"\nAnalyzed {processed_count} images from {len(all_landmarks)} people")
    print("\nYour Dataset's Optimal Face Template (normalized coordinates):")
    print("Left Eye:    [{:.8f}, {:.8f}]".format(dataset_template[0][0], dataset_template[0][1]))
    print("Right Eye:   [{:.8f}, {:.8f}]".format(dataset_template[1][0], dataset_template[1][1]))
    print("Nose Tip:    [{:.8f}, {:.8f}]".format(dataset_template[2][0], dataset_template[2][1]))
    print("Left Mouth:  [{:.8f}, {:.8f}]".format(dataset_template[3][0], dataset_template[3][1]))
    print("Right Mouth: [{:.8f}, {:.8f}]".format(dataset_template[4][0], dataset_template[4][1]))
    
    print("\nStandard Deviations:")
    for i, name in enumerate(['Left Eye', 'Right Eye', 'Nose Tip', 'Left Mouth', 'Right Mouth']):
        print(f"{name}: [{dataset_std[i][0]:.6f}, {dataset_std[i][1]:.6f}]")
    
    print("\nComparison with Generic Template:")
    generic_template = np.array([
        [0.31556875, 0.4615741],   # Left eye
        [0.68262106, 0.4615741],   # Right eye  
        [0.5, 0.61041667],         # Nose tip
        [0.34947552, 0.84090909],  # Left mouth
        [0.65052448, 0.84090909]   # Right mouth
    ])
    
    differences = np.abs(dataset_template - generic_template)
    print("Absolute differences:")
    for i, name in enumerate(['Left Eye', 'Right Eye', 'Nose Tip', 'Left Mouth', 'Right Mouth']):
        print(f"{name}: [{dataset_std[i][0]:.6f}, {dataset_std[i][1]:.6f}]")

    
    max_diff = np.max(differences)
    print(f"\nMaximum difference: {max_diff:.6f}")
    
    if max_diff > 0.05:  # 5% difference
        print("⚠️  SIGNIFICANT DIFFERENCES FOUND!")
        print("Your dataset needs a custom template for optimal alignment.")
        return dataset_template
    else:
        print("✅ Generic template is close enough to your dataset.")
        return None

def visualize_template_comparison(dataset_template):
    """Visualize the difference between templates"""
    if dataset_template is None:
        return
        
    generic_template = np.array([
        [0.31556875, 0.4615741],
        [0.68262106, 0.4615741], 
        [0.5, 0.61041667],
        [0.34947552, 0.84090909],
        [0.65052448, 0.84090909]
    ])
    
    plt.figure(figsize=(12, 6))
    
    # Plot both templates
    plt.subplot(1, 2, 1)
    plt.title("Template Comparison")
    plt.scatter(generic_template[:, 0], generic_template[:, 1], c='red', s=100, label='Generic', marker='x')
    plt.scatter(dataset_template[:, 0], dataset_template[:, 1], c='blue', s=100, label='Your Dataset', marker='o')
    
    # Connect corresponding points
    for i in range(5):
        plt.plot([generic_template[i, 0], dataset_template[i, 0]], 
                [generic_template[i, 1], dataset_template[i, 1]], 'gray', alpha=0.5)
    
    plt.legend()
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.grid(True, alpha=0.3)
    
    # Plot difference magnitudes
    plt.subplot(1, 2, 2)
    differences = np.linalg.norm(dataset_template - generic_template, axis=1)
    landmarks = ['Left Eye', 'Right Eye', 'Nose', 'L Mouth', 'R Mouth']
    plt.bar(landmarks, differences)
    plt.title("Difference Magnitudes")
    plt.ylabel("Euclidean Distance")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("template_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Template comparison saved as 'template_analysis.png'")

if __name__ == "__main__":
    print("🔍 Analyzing your dataset for optimal face alignment template...")
    dataset_template = analyze_dataset_landmarks()
    
    if dataset_template is not None:
        print("\n📊 Generating visualization...")
        visualize_template_comparison(dataset_template)
        
        print("\n💾 Generating updated code with your optimal template...")
        print("\nAdd this template to your alignment function:")
        print("template = np.array([")
        for i, (x, y) in enumerate(dataset_template):
            name = ['# Left eye', '# Right eye', '# Nose tip', '# Left mouth', '# Right mouth'][i]
            print(f"    [{x:.8f}, {y:.8f}],   {name}")
        print("], dtype=np.float32)")
    else:
        print("\n✅ The generic template should work fine for your dataset.")
        print("You can proceed with the current alignment code.")
import os
import cv2
import torch
import numpy as np
from deepface import DeepFace
from torchvision import transforms
from PIL import Image
from models.model import PixelWise
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def detect_and_crop_face(frame):
    faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
    if not faces:
        return None
    faces_sorted = sorted(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'], reverse=True)
    largest_face = faces_sorted[0]['face']
    if largest_face is None or largest_face.shape[0] == 0 or largest_face.shape[1] == 0:
        return None
    if largest_face.max() <= 1:
        largest_face = (largest_face * 224).astype(np.uint8)
    largest_face = cv2.resize(largest_face, (224, 224), interpolation=cv2.INTER_AREA)
    if len(largest_face.shape) == 3 and largest_face.shape[2] == 3:
        largest_face = cv2.cvtColor(largest_face, cv2.COLOR_BGR2RGB)
    return largest_face

def load_model(ckpt_path, device):
    model = PixelWise(pretrained=True).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model

def get_predict(mask, threshold):
    with torch.no_grad():
        score = torch.mean(mask, dim=(1, 2, 3)) if mask.dim() == 4 else torch.mean(mask, dim=(1, 2))
        return (score > threshold).float(), score

def predict(frame, model, transform, device, threshold):
    face = detect_and_crop_face(frame)
    if face is None:
        return None, None
    image = transform(Image.fromarray(face)).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred, _ = model(image)
        preds, score = get_predict(mask_pred, threshold)
        return preds.item(), score.item()

if __name__ == "__main__":
    # Spoof
    image_path = "frame016.jpg" 
    
    #Live
    # image_path = "image.png" 
    
    ckpt_path = "best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    model = load_model(ckpt_path, device)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
    else:
        for threshold in np.arange(0.5, 1.0, 0.05):
            label, score = predict(image, model, transform, device, threshold)
            result = "No face detected" if label is None else ("Live" if label == 1 else "Spoof")
            print(f"Threshold: {threshold:.2f} | Score: {score:.4f} | Result: {result}")
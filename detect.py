import os
import torch
import cv2
from deepface import DeepFace
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from models.model import PixelWise

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def detect_and_crop_face(img_path):
    """
    Detect the largest face in the image using RetinaFace and return the cropped face.
    Ensures the face is in the correct RGB format and resized to 224x224.
    Returns None if no face is detected.
    """
    faces = DeepFace.extract_faces(img_path, detector_backend='retinaface', enforce_detection=False)
    if len(faces) == 0:
        print(f"No face detected in {img_path}")
        return None

    # Sort faces by area (largest face first)
    faces_sorted = sorted(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'], reverse=True)
    largest_face = faces_sorted[0]['face']

    if largest_face is None or largest_face.shape[0] == 0 or largest_face.shape[1] == 0:
        print(f"Invalid face dimensions in {img_path}")
        return None

    if largest_face.max() <= 1:  # Normalize if necessary
        largest_face = (largest_face * 255).astype(np.uint8)  # Changed 224 to 255 for proper scaling

    largest_face = cv2.resize(largest_face, (224, 224), interpolation=cv2.INTER_AREA)

    if len(largest_face.shape) == 3 and largest_face.shape[2] == 3:  # Ensure RGB format
        largest_face = cv2.cvtColor(largest_face, cv2.COLOR_BGR2RGB)

    return largest_face

def load_model(checkpoint_path, device):
    """
    Load the PixelWise model from a checkpoint for inference.
    If quantized keys are detected, convert the model to a quantized version.
    """
    # Load checkpoint and extract state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Check if the state dict is quantized (look for typical quantized keys)
    if any("scale" in k for k in state_dict.keys()):
        print("Detected quantized model state_dict. Converting model for quantized inference.")
        # Instantiate the FP32 model without pretrained weights
        model_fp32 = PixelWise(pretrained=False).to(device)
        model_fp32.eval()
        # Convert the model to a quantized version
        model = torch.quantization.convert(model_fp32, inplace=False)
    else:
        model = PixelWise(pretrained=True).to(device)
    
    # Load state dict (use strict=False to account for any key differences)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

def get_predict(mask, label, threshold=0.6, score_type='combined'):
    with torch.no_grad():
        if score_type == 'pixel':
            score = torch.mean(mask, dim=(1, 2, 3))  # Changed 'axis' to 'dim' for PyTorch
        elif score_type == 'binary':
            score = label
        else:
            score = (torch.mean(mask, dim=(1, 2, 3)) + label) / 2
        preds = (score > threshold).to(torch.uint8).cpu()  # Changed uint16 to uint8 for simplicity
        print(f"preds from get_predict: {preds}")
        return preds, score

def predict(image_path, model, transform, device):
    """
    Detect, crop, and predict the label for an image.
    """
    cropped_face = detect_and_crop_face(image_path)
    if cropped_face is None:
        print(f"Skipping {image_path} due to no detected face.")
        return None, None, None

    image = transform(Image.fromarray(cropped_face)).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_pred, label_pred = model(image)
        preds, score = get_predict(mask_pred, label_pred)
        predicted_label = preds.item()
    return predicted_label, mask_pred.squeeze(), score

if __name__ == "__main__":
    # Define checkpoint path and single image path
    checkpoint_path = "quantized_model.pth"
    image_path = "image1.png"  # specify the image path to process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the same transform as training for consistency
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the model
    model = load_model(checkpoint_path, device)

    # Process the single image and output prediction along with the score
    predicted_label, mask, score = predict(image_path, model, transform, device)
    if predicted_label is not None:
        label_text = "Live" if predicted_label == 1 else "Spoof"
        print(f"Prediction for {image_path}: {label_text}")
        print(f"Score: {score.item() if score.numel() == 1 else score}")
    else:
        print(f"Skipping {image_path} due to no detected face.")
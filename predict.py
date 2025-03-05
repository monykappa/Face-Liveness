import os
import time
import torch
import cv2
from deepface import DeepFace
from torchvision import transforms
from PIL import Image
import numpy as np
from models.model import PixelWise
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Function to ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create detection type folders
def create_detection_folders(base_dir):
    live_dir = os.path.join(base_dir, "live")
    spoof_dir = os.path.join(base_dir, "spoof")
    ensure_dir(live_dir)
    ensure_dir(spoof_dir)
    return live_dir, spoof_dir

def detect_and_crop_face(img, return_bbox=False):
    """
    Detect and crop the largest face in an image.
    
    Args:
        img: Input image (numpy array)
        return_bbox: Whether to return the bounding box coordinates
        
    Returns:
        Cropped face image or (cropped_face, bbox) if return_bbox=True
    """
    try:
        faces = DeepFace.extract_faces(img, detector_backend='ssd', enforce_detection=False)
        
        if not faces:
            return None if not return_bbox else (None, None)
            
        faces_sorted = sorted(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'], reverse=True)
        largest_face = faces_sorted[0]['face']
        bbox = faces_sorted[0]['facial_area']
        
        # Validation checks
        if largest_face is None or largest_face.shape[0] == 0 or largest_face.shape[1] == 0:
            return None if not return_bbox else (None, None)
            
        # Normalize and resize
        if largest_face.max() <= 1:
            largest_face = (largest_face * 255).astype(np.uint8)
            
        largest_face = cv2.resize(largest_face, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Ensure correct color format
        if len(largest_face.shape) == 3 and largest_face.shape[2] == 3:
            largest_face = cv2.cvtColor(largest_face, cv2.COLOR_BGR2RGB)
            
        if return_bbox:
            return largest_face, (bbox['x'], bbox['y'], bbox['w'], bbox['h'])
            
        return largest_face
        
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return None if not return_bbox else (None, None)

def load_model(checkpoint_path, device):
    model = PixelWise(pretrained=True).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model

def get_predict(mask, label, threshold=0.7, score_type='combined'):
    with torch.no_grad():
        if score_type == 'pixel':
            score = torch.mean(mask, axis=(1,2,3))
        elif score_type == 'binary':
            score = label
        else:
            score = (torch.mean(mask, axis=(1,2,3)) + label) / 2
        preds = (score > threshold).to(torch.uint16).cpu()
        return preds, score

def predict_frame(frame, model, transform, device, save_dirs=None):
    """
    Process a frame to detect faces and predict if it's real or spoof.
    
    Args:
        frame: Input video frame
        model: The loaded PixelWise model
        transform: Image transformations to apply
        device: Computing device (CPU/GPU)
        save_dirs: Tuple of (live_dir, spoof_dir) or None to disable saving
        
    Returns:
        (prediction, mask, detection_status, saved_path, bbox)
    """
    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
    
    # Get both the cropped face and bounding box
    result = detect_and_crop_face(small_frame, return_bbox=True)
    
    # Check if face detection was successful
    if result is None or result[0] is None:
        return None, None, "No Face Detected", None, None
        
    cropped_face, small_bbox = result
    
    # Scale bounding box back to original frame size
    if small_bbox:
        x, y, w, h = small_bbox
        bbox = (int(x*2), int(y*2), int(w*2), int(h*2))
    else:
        bbox = None
    
    # Model prediction
    image = transform(Image.fromarray(cropped_face)).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred, label_pred = model(image)
        preds, _ = get_predict(mask_pred, label_pred)
        predicted_label = preds.item()
    
    # Save the face to appropriate folder if directories are provided
    saved_path = None
    if save_dirs and cropped_face is not None:
        live_dir, spoof_dir = save_dirs
        # Choose directory based on prediction
        save_dir = live_dir if predicted_label == 1 else spoof_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"face_{timestamp}.jpg"
        saved_path = os.path.join(save_dir, filename)
        
        # Save the detected face
        cv2.imwrite(saved_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
    
    return predicted_label, mask_pred.squeeze(), "Face Detected", saved_path, bbox

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Set up face logging
    save_faces = True  # Set to False if you don't want to save faces
    base_dir = "detected_faces" if save_faces else None
    save_dirs = None
    
    if base_dir:
        ensure_dir(base_dir)
        live_dir, spoof_dir = create_detection_folders(base_dir)
        save_dirs = (live_dir, spoof_dir)
        print(f"Detected faces will be saved to:")
        print(f"  - Live faces: {os.path.abspath(live_dir)}")
        print(f"  - Spoof faces: {os.path.abspath(spoof_dir)}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Frame processing settings
    process_every_n_frames = 3
    frame_count = 0
    last_prediction = None
    last_mask = None
    last_detection_status = "Starting..."
    last_bbox = None
    saved_folder_type = None
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # For calculating FPS
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Only process every nth frame
        if frame_count % process_every_n_frames == 0:
            result = predict_frame(frame, model, transform, device, save_dirs)
            
            if result[0] is not None:
                last_prediction, last_mask, last_detection_status, saved_path, last_bbox = result
                if saved_path:
                    saved_folder_type = "live" if last_prediction == 1 else "spoof"
        
        frame_count += 1
        
        # Use the last valid prediction for display
        if last_prediction is not None:
            # Draw bounding box if available
            if last_bbox:
                x, y, w, h = last_bbox
                color = (0, 255, 0) if last_prediction == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Show prediction label
            label_text = "Live" if last_prediction == 1 else "Spoof"
            color = (0, 255, 0) if last_prediction == 1 else (0, 0, 255)
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw detection status
        cv2.putText(frame, last_detection_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show FPS
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show logging status and saved folder info
        if save_dirs:
            cv2.putText(frame, "Logging: ON", (frame.shape[1] - 120, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if saved_folder_type:
                save_info = f"Saved to: {saved_folder_type}"
                save_color = (0, 255, 0) if saved_folder_type == "live" else (0, 0, 255)
                cv2.putText(frame, save_info, (frame.shape[1] - 180, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, save_color, 2)
        
        # Display the frame
        cv2.imshow("Realtime Detection", frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):  # Toggle logging
            save_faces = not save_faces
            if save_faces:
                base_dir = "detected_faces"
                ensure_dir(base_dir)
                live_dir, spoof_dir = create_detection_folders(base_dir)
                save_dirs = (live_dir, spoof_dir)
                print(f"Logging enabled. Faces saved to:")
                print(f"  - Live faces: {os.path.abspath(live_dir)}")
                print(f"  - Spoof faces: {os.path.abspath(spoof_dir)}")
            else:
                save_dirs = None
                print("Logging disabled")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

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
from collections import deque
import csv

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

#############################################
#    Utility Functions and Directories      #
#############################################

def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_detection_folders(base_dir):
    """Create sub-folders for live and spoof images."""
    live_dir = os.path.join(base_dir, "live")
    spoof_dir = os.path.join(base_dir, "spoof")
    ensure_dir(live_dir)
    ensure_dir(spoof_dir)
    return live_dir, spoof_dir

#############################################
#         Face Detection Function           #
#############################################

def detect_and_crop_face(img, return_bbox=False, for_display=False):
    """
    Detect and crop the largest face in an image using DeepFace.
    
    Args:
        img: Input image (numpy array)
        return_bbox: Whether to return bounding box coordinates.
        for_display: If True, keeps BGR format for display, otherwise converts to RGB for model.
        
    Returns:
        Cropped face image or (cropped_face, bbox) if return_bbox=True.
    """
    try:
        faces = DeepFace.extract_faces(img, detector_backend='ssd', enforce_detection=False)
        if not faces:
            return None if not return_bbox else (None, None)
            
        # Sort faces by area (largest first)
        faces_sorted = sorted(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'], 
                              reverse=True)
        largest_face = faces_sorted[0]['face']
        bbox = faces_sorted[0]['facial_area']
        
        # Validate face
        if largest_face is None or largest_face.shape[0] == 0 or largest_face.shape[1] == 0:
            return None if not return_bbox else (None, None)
            
        # Normalize if image is float [0,1]
        if largest_face.max() <= 1:
            largest_face = (largest_face * 255).astype(np.uint8)
            
        # Resize to model input size (224 x 224)
        largest_face = cv2.resize(largest_face, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB if needed (for model processing but not for display)
        if len(largest_face.shape) == 3 and largest_face.shape[2] == 3 and not for_display:
            largest_face = cv2.cvtColor(largest_face, cv2.COLOR_BGR2RGB)
            
        if return_bbox:
            return largest_face, (bbox['x'], bbox['y'], bbox['w'], bbox['h'])
        return largest_face
        
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return None if not return_bbox else (None, None)

#############################################
#         Image Quality Assessment          #
#############################################

def assess_image_quality(image):
    """
    Assess the quality of an image to determine if it's suitable for liveness detection.
    Returns a score between 0 (poor) and 1 (excellent) and quality metrics.
    """
    # Convert to grayscale for calculations
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    metrics = {}
    
    # 1. Sharpness/blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize, cap at 1.0
    metrics['sharpness'] = sharpness_score
    
    # 2. Brightness assessment
    brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - 2.0 * abs(brightness - 0.5)  # Penalize too dark or too bright
    metrics['brightness'] = brightness_score
    
    # 3. Contrast assessment
    contrast = np.std(gray.astype(np.float32)) / 128.0
    contrast_score = min(contrast, 1.0)
    metrics['contrast'] = contrast_score
    
    # Calculate weighted final score
    final_score = (0.5 * sharpness_score +  # Sharpness is most important
                  0.25 * brightness_score +
                  0.25 * contrast_score)
    
    return final_score, metrics

#############################################
#           Model Loading & Prediction      #
#############################################

def load_model(checkpoint_path, device):
    """
    Load the PixelWise model from checkpoint.
    """
    model = PixelWise(pretrained=True).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model

def get_predict(mask, label, threshold=0.7, score_type='combined'):
    """
    Compute prediction score and binary decision based on mask and label.
    """
    with torch.no_grad():
        if score_type == 'pixel':
            score = torch.mean(mask, axis=(1, 2, 3))
        elif score_type == 'binary':
            score = label
        else:  # combined
            score = (torch.mean(mask, axis=(1, 2, 3)) + label) / 2
        preds = (score > threshold).to(torch.uint16).cpu()
        return preds, score

def predict_frame(frame, model, transform, device, threshold=0.7):
    """
    Process a frame to detect face and predict spoof vs. live using a specific threshold.
    """
    # Get a face for display (keeping BGR format)
    display_result = detect_and_crop_face(frame, return_bbox=True, for_display=True)
    display_face = None
    if display_result and display_result[0] is not None:
        display_face, _ = display_result
    
    # Get a face for model processing (converted to RGB format)
    result = detect_and_crop_face(frame, return_bbox=True)
    if result is None or result[0] is None:
        return None, None, None, None, None, None, None
        
    cropped_face, bbox = result
    
    # Assess face image quality
    quality_score, quality_metrics = assess_image_quality(cropped_face)
    
    # Preprocess: Transform for model input
    image = transform(Image.fromarray(cropped_face)).unsqueeze(0).to(device)
    
    # Forward pass through model
    with torch.no_grad():
        mask_pred, label_pred = model(image)
        preds, score_tensor = get_predict(mask_pred, label_pred, threshold=threshold)
        score_value = score_tensor.item()
        predicted_label = preds.item()
    
    return predicted_label, score_value, mask_pred.squeeze(), cropped_face, bbox, quality_score, display_face

#############################################
#        Multi-Threshold Manager            #
#############################################

class MultiThresholdManager:
    def __init__(self, model, transform, device, thresholds):
        """
        Manages multiple threshold evaluations simultaneously.
        
        Args:
            model: The loaded PixelWise model
            transform: Image transformation pipeline
            device: CPU or GPU device
            thresholds: List of threshold values to evaluate
        """
        self.model = model
        self.transform = transform
        self.device = device
        self.thresholds = thresholds
        
        # Create separate save directories for each threshold
        self.base_dirs = {}
        self.save_dirs = {}
        self.frame_counters = {t: 0 for t in thresholds}
        self.results_files = {}
        
        # Create directory structure
        for threshold in thresholds:
            threshold_str = f"{threshold:.2f}".replace('.', '_')
            base_dir = f"output/threshold_{threshold_str}"
            ensure_dir(base_dir)
            self.base_dirs[threshold] = base_dir
            
            live_dir, spoof_dir = create_detection_folders(base_dir)
            self.save_dirs[threshold] = (live_dir, spoof_dir)
            
            # Create results file for this threshold
            results_file = os.path.join(base_dir, "results.csv")
            self.results_files[threshold] = results_file
            self._initialize_results_file(results_file)
            
            print(f"Threshold {threshold}: Results will be saved to {base_dir}")
    
    def _initialize_results_file(self, file_path):
        """Create and initialize a results log file with headers"""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'prediction', 'score', 'quality_score', 'timestamp'])
    
    def _log_result(self, threshold, image_path, prediction, score, quality):
        """Log a detection result to the CSV file for the specific threshold"""
        results_file = self.results_files[threshold]
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([image_path, 
                            'live' if prediction == 1 else 'spoof', 
                            f"{score:.4f}", 
                            f"{quality:.4f}", 
                            timestamp])
    
    def process_frame(self, frame):
        """
        Process a single frame with all thresholds.
        
        Returns:
            Dictionary of results keyed by threshold value
        """
        # Detect and crop face once (shared across all thresholds)
        result = predict_frame(frame, self.model, self.transform, self.device)
        
        if result[0] is None:  # No face detected
            return None
        
        # Unpack results
        _, score_value, mask, cropped_face_rgb, bbox, quality_score, display_face = result
        
        # Process with each threshold
        threshold_results = {}
        
        for threshold in self.thresholds:
            # Determine prediction based on this threshold
            predicted_label = 1 if score_value > threshold else 0
            
            # Save the image to the appropriate threshold directory
            live_dir, spoof_dir = self.save_dirs[threshold]
            save_dir = live_dir if predicted_label == 1 else spoof_dir
            
            # Sequential frame numbering for this threshold
            frame_counter = self.frame_counters[threshold]
            filename = f"frame{frame_counter:03d}.jpg"
            saved_path = os.path.join(save_dir, filename)
            
            # Convert RGB back to BGR for saving
            cv2.imwrite(saved_path, cv2.cvtColor(cropped_face_rgb, cv2.COLOR_RGB2BGR))
            
            # Log the result
            self._log_result(threshold, saved_path, predicted_label, score_value, quality_score)
            
            # Increment the frame counter for this threshold
            self.frame_counters[threshold] += 1
            
            # Store results for this threshold
            threshold_results[threshold] = {
                'label': predicted_label,
                'score': score_value,
                'path': saved_path,
                'quality': quality_score
            }
        
        return {
            'bbox': bbox,
            'display_face': display_face,
            'threshold_results': threshold_results,
            'mask': mask,
            'quality_score': quality_score
        }

#############################################
#                Main Application           #
#############################################

if __name__ == "__main__":
    # Configuration & Initialization
    checkpoint_path = "best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Define thresholds to evaluate
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Load the trained model
    model = load_model(checkpoint_path, device)
    
    # Initialize multi-threshold manager
    threshold_manager = MultiThresholdManager(model, transform, device, thresholds)

    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # Frame processing control
    process_every_n_frames = 10  # Process every 10th frame to avoid too many files
    frame_count = 0
    
    # Motion detection for frame selection
    prev_frame = None
    skip_frame = True  # Skip first frame
    
    # Initialize results variable
    results = None

    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
        exit(0)
    
    print(f"Testing {len(thresholds)} thresholds: {thresholds}")
    
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
        
        # Process only every n-th frame and when motion is low
        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            # Skip this frame but still display with previous results
            pass
        else:
            # Check for motion
            if prev_frame is not None:
                gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = np.mean(cv2.absdiff(gray1, gray2)) / 255.0
                skip_frame = motion > 0.05  # Skip if too much motion
            
            prev_frame = frame.copy()
            
            # Process frame if not skipping
            if not skip_frame:
                results = threshold_manager.process_frame(frame)
            else:
                results = None
        
        # Display results
        if results:
            bbox = results['bbox']
            display_face = results['display_face']
            threshold_results = results['threshold_results']
            quality_score = results['quality_score']
            
            # Draw bounding box
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
            # Show the detected face in the top right corner
            if display_face is not None:
                h, w = frame.shape[:2]
                face_size = display_face.shape[0]  # Face should be square (224x224)
                
                # Position in top-right corner with padding
                pad = 10
                x_offset = w - face_size - pad
                y_offset = pad
                
                # Create a region of interest
                roi = frame[y_offset:y_offset+face_size, x_offset:x_offset+face_size]
                
                # Only overlay if ROI is the right size
                if roi.shape == display_face.shape:
                    frame[y_offset:y_offset+face_size, x_offset:x_offset+face_size] = display_face
                
                # Draw a border around the preview
                cv2.rectangle(frame, 
                            (x_offset-1, y_offset-1), 
                            (x_offset+face_size+1, y_offset+face_size+1), 
                            (255, 255, 255), 1)
            
            # Display the score and prediction for each threshold
            y_pos = 30
            for i, threshold in enumerate(sorted(thresholds)):
                result = threshold_results[threshold]
                label = "Live" if result['label'] == 1 else "Spoof"
                score = result['score']
                color = (0, 255, 0) if result['label'] == 1 else (0, 0, 255)
                
                # Display threshold results
                threshold_text = f"T={threshold:.2f}: {label} ({score:.2f})"
                cv2.putText(frame, threshold_text, (10, y_pos + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show quality score
            cv2.putText(frame, f"Quality: {quality_score:.2f}", 
                       (10, y_pos + len(thresholds)*30 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show the video feed with overlays
        cv2.imshow("Multi-Threshold Liveness Detection", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            # Force capture on next frame
            skip_frame = False
            frame_count = process_every_n_frames - 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
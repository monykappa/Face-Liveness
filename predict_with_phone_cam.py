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
    
    # 4. Face size relative to image (if bounding box available)
    # This is handled by the face detection function already
    
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

def predict_frame(frame, model, transform, device, save_dirs=None, threshold=0.7, frame_counter=0):
    """
    Process a frame to detect face and predict spoof vs. live.
    """
    # Get a face for display (keeping BGR format)
    display_result = detect_and_crop_face(frame, return_bbox=True, for_display=True)
    display_face = None
    if display_result and display_result[0] is not None:
        display_face, _ = display_result
    
    # Get a face for model processing (converted to RGB format)
    result = detect_and_crop_face(frame, return_bbox=True)
    if result is None or result[0] is None:
        return None, None, None, "No Face Detected", None, None, None, display_face
        
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
    
    # Optionally log/save the detected face
    saved_path = None
    if save_dirs and cropped_face is not None:
        live_dir, spoof_dir = save_dirs
        save_dir = live_dir if predicted_label == 1 else spoof_dir
        
        # Sequential frame numbering
        filename = f"frame{frame_counter:03d}.jpg"
        saved_path = os.path.join(save_dir, filename)
        cv2.imwrite(saved_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
    
    return predicted_label, score_value, mask_pred.squeeze(), "Face Detected", saved_path, bbox, quality_score, display_face

#############################################
#        Batch Processing Class             #
#############################################

class LivenessDetector:
    def __init__(self, model, transform, device, threshold=0.7):
        self.model = model
        self.transform = transform
        self.device = device
        self.threshold = threshold
        
        # Batch processing parameters
        self.batch_size = 3  # Number of high-quality frames to collect before decision
        self.min_quality_threshold = 0.6  # Minimum quality score to consider a frame
        self.collected_frames = []
        self.quality_scores = []
        
        # Stabilization parameters
        self.result_history = deque(maxlen=10)  # Store recent prediction results
        self.stable_prediction = None  # Current stable prediction (weighted majority)
        self.confidence_score = 0.0    # Confidence in current prediction
        
        # Frame capturing control
        self.frames_since_last_capture = 0
        self.min_frames_between_captures = 5  # Minimum frames to wait before capturing again
        
        # Motion detection params
        self.prev_frame = None
        self.motion_score = 0.0
        
        # Sequential frame numbering
        self.frame_counter = 0
        
        # For display and visualization
        self.display_frame = None
        self.display_bbox = None
        
        # Create results log file
        self.results_file = "liveness_results.csv"
        self._initialize_results_file()
        
    def detect_motion(self, frame):
        """Calculate motion amount between consecutive frames"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference and normalize
        frame_diff = cv2.absdiff(gray, self.prev_frame)
        motion_score = np.mean(frame_diff) / 255.0
        
        self.prev_frame = gray
        self.motion_score = motion_score
        
        return motion_score
    
    def should_capture_frame(self, frame):
        """Determine if we should capture this frame for liveness detection"""
        # Check if minimum frames have passed
        if self.frames_since_last_capture < self.min_frames_between_captures:
            self.frames_since_last_capture += 1
            return False
        
        # Check motion level - capture when motion is low
        motion = self.detect_motion(frame)
        should_capture = motion < 0.05  # Low motion threshold
        
        if should_capture:
            self.frames_since_last_capture = 0
            
        return should_capture
    
    def process_frame(self, frame, save_dirs=None):
        """
        Process a video frame, collecting high-quality frames for batch prediction.
        Returns prediction results and bbox for display.
        """
        # Check if we should capture this frame
        if not self.should_capture_frame(frame):
            # Just return the current stable prediction without processing
            return (self.stable_prediction, self.confidence_score, None, 
                    "Waiting for stable face...", None, None, None, 
                    self.motion_score, self.display_frame)
        
        # Process the frame
        result = predict_frame(frame, self.model, self.transform, 
                              self.device, save_dirs, self.threshold, self.frame_counter)
        
        if result[0] is None:  # No face detected
            self.display_frame = None
            return (*result, self.motion_score, None)
        
        pred_label, score_value, mask, status, saved_path, bbox, quality_score, display_face = result
        
        # Store the display face for visualization
        self.display_frame = display_face
        self.display_bbox = bbox
        
        # If a face was saved, increment the frame counter and log the result
        if saved_path:
            # Log detection result
            self._log_result(saved_path, pred_label, score_value, quality_score)
            # Increment frame counter for next save
            self.frame_counter += 1
        
        # If quality is good enough, add to collected frames
        if quality_score >= self.min_quality_threshold:
            self.collected_frames.append((pred_label, score_value))
            self.quality_scores.append(quality_score)
            
            # If we have enough frames, make a batch decision
            if len(self.collected_frames) >= self.batch_size:
                self._make_batch_decision()
        
        return (self.stable_prediction, self.confidence_score, mask, 
                f"Quality: {quality_score:.2f}, Motion: {self.motion_score:.2f}",
                saved_path, bbox, quality_score, self.motion_score, display_face)
    
    def _initialize_results_file(self):
        """Create and initialize the results log file with headers"""
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'prediction', 'confidence_score', 'quality_score', 'timestamp'])
    
    def _log_result(self, image_path, prediction, confidence, quality):
        """Log a detection result to the CSV file"""
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([image_path, 
                            'live' if prediction == 1 else 'spoof', 
                            f"{confidence:.4f}", 
                            f"{quality:.4f}", 
                            timestamp])
    
    def _make_batch_decision(self):
        """
        Process collected frames to make a weighted batch decision,
        giving more weight to higher quality frames.
        """
        if not self.collected_frames:
            return
        
        # Weight predictions by quality score
        live_weight = 0
        spoof_weight = 0
        total_quality = sum(self.quality_scores)
        
        for (label, score), quality in zip(self.collected_frames, self.quality_scores):
            weight = quality / total_quality if total_quality > 0 else 1.0 / len(self.collected_frames)
            if label == 1:  # Live
                live_weight += weight
            else:  # Spoof
                spoof_weight += weight
        
        # Determine final prediction
        new_prediction = 1 if live_weight > spoof_weight else 0
        confidence = max(live_weight, spoof_weight)
        
        # Add to history
        self.result_history.append((new_prediction, confidence))
        
        # Update stable prediction with time decay weighting
        recent_results = list(self.result_history)
        if recent_results:
            # Weight recent results more heavily
            weights = [0.6**i for i in range(len(recent_results))]
            
            # Normalize weights
            total_weight = sum(weights)
            norm_weights = [w/total_weight for w in weights]
            
            # Calculate weighted prediction
            live_prob = sum(norm_weights[i] * (1 if pred == 1 else 0)
                           for i, (pred, _) in enumerate(recent_results))
            
            # Update stable prediction and confidence
            self.stable_prediction = 1 if live_prob > 0.5 else 0
            self.confidence_score = abs(live_prob - 0.5) * 2  # Scale to [0,1]
        
        # Reset collected frames
        self.collected_frames = []
        self.quality_scores = []

#############################################
#                Main Application           #
#############################################

if __name__ == "__main__":
    # Configuration & Initialization
    checkpoint_path = "quantized_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Logging configuration for saving face images
    save_faces = True  # Set to False to disable logging
    base_dir = "detected_frames" if save_faces else None
    save_dirs = None

    if base_dir:
        ensure_dir(base_dir)
        live_dir, spoof_dir = create_detection_folders(base_dir)
        save_dirs = (live_dir, spoof_dir)
        print("Detected faces will be saved to:")
        print(f"  Live: {os.path.abspath(live_dir)}")
        print(f"  Spoof: {os.path.abspath(spoof_dir)}")
        print(f"Results will be logged to: liveness_results.csv")
    
    # Load the trained model
    model = load_model(checkpoint_path, device)
    
    # Initialize liveness detector
    detector = LivenessDetector(model, transform, device)

    # FPS calculation variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    # Camera selection options
    use_phone_camera = True  # Set to True to use phone camera, False for webcam
    
    if use_phone_camera:
        # Replace with your phone's IP camera URL
        # Format example for IP Webcam app: http://192.168.1.100:8080/video
        phone_ip = "172.17.199.53"  # Replace with your phone's IP address
        phone_port = "8080"         # Default port for IP Webcam app
        camera_url = f"http://{phone_ip}:{phone_port}/video"
        
        print(f"Connecting to phone camera at: {camera_url}")
        print("(If connection fails, make sure phone and computer are on the same WiFi network)")
        print("(Also verify the IP address and port are correct)")
        
        # Initialize phone camera
        cap = cv2.VideoCapture(camera_url)
        
        # For some IP camera apps, you might need to use MJPEG stream instead:
        # cap = cv2.VideoCapture(f"http://{phone_ip}:{phone_port}/videofeed")
    else:
        # Use computer webcam
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera not accessible. Check the connection or URL.")
        exit(0)
    
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
        
        # Process frame with the liveness detector
        smoothed_label, confidence, mask, status, saved_path, bbox, quality, motion, display_face = \
            detector.process_frame(frame, save_dirs)
        
        # Draw bounding box and prediction overlay
        if bbox:
            x, y, w, h = bbox
            color = (0, 255, 0) if smoothed_label == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Show the detected face in its original colors (if available)
        if display_face is not None:
            # Display the face in the corner of the frame (for debugging)
            h, w = frame.shape[:2]
            face_size = display_face.shape[0]  # Assuming square face
            
            # Position the face in the top-right corner with padding
            pad = 10
            x_offset = w - face_size - pad
            y_offset = pad
            
            # Create a region of interest
            roi = frame[y_offset:y_offset+face_size, x_offset:x_offset+face_size]
            
            # Only overlay if ROI is the right size
            if roi.shape == display_face.shape:
                frame[y_offset:y_offset+face_size, x_offset:x_offset+face_size] = display_face
            
            # Draw a border around the face
            cv2.rectangle(frame, 
                        (x_offset-1, y_offset-1), 
                        (x_offset+face_size+1, y_offset+face_size+1), 
                        (255, 255, 255), 1)
        
        # Display prediction and confidence
        label_text = "Live" if smoothed_label == 1 else "Spoof" if smoothed_label == 0 else "Waiting..."
        
        if smoothed_label is not None:
            conf_text = f" ({confidence:.2f})"
            label_text += conf_text
            
        color = (0, 255, 0) if smoothed_label == 1 else (0, 0, 255) if smoothed_label == 0 else (255, 255, 255)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show status information
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # If logging is enabled, show which folder the current face was saved to
        if save_dirs and saved_path:
            folder_type = "live" if "live" in saved_path else "spoof"
            frame_number = os.path.basename(saved_path).replace(".jpg", "")
            save_info = f"Saved: {frame_number} ({folder_type})"
            save_color = (0, 255, 0) if folder_type == "live" else (0, 0, 255)
            cv2.putText(frame, save_info, (frame.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, save_color, 2)
        
        # Show the video feed with overlays
        cv2.imshow("Improved Liveness Detection", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            # Toggle logging
            save_faces = not save_faces
            if save_faces:
                base_dir = "detected_faces_batch"
                ensure_dir(base_dir)
                live_dir, spoof_dir = create_detection_folders(base_dir)
                save_dirs = (live_dir, spoof_dir)
                print("Logging enabled. Faces saved to:")
                print(f"  Live: {os.path.abspath(live_dir)}")
                print(f"  Spoof: {os.path.abspath(spoof_dir)}")
            else:
                save_dirs = None
                print("Logging disabled")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
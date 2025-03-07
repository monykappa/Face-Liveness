import os
import time
import torch
import cv2
import warnings
import numpy as np
import csv
from datetime import datetime
from deepface import DeepFace
from torchvision import transforms
from PIL import Image
from models.model import PixelWise

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_detection_folders(base_dir):
    live_dir = os.path.join(base_dir, "live")
    spoof_dir = os.path.join(base_dir, "spoof")
    ensure_dir(live_dir)
    ensure_dir(spoof_dir)
    return live_dir, spoof_dir

class MultiThresholdManager:
    def __init__(self, model, transform, device, thresholds, output_folder):
        self.model = model
        self.transform = transform
        self.device = device
        self.thresholds = thresholds
        self.output_folder = output_folder
        self.base_dirs = {}
        self.save_dirs = {}
        self.frame_counters = {t: 0 for t in thresholds}
        self.results_files = {}
        for t in thresholds:
            base_dir = os.path.join(self.output_folder, f"threshold_{t:.2f}".replace('.', '_'))
            ensure_dir(base_dir)
            self.base_dirs[t] = base_dir
            live, spoof = create_detection_folders(base_dir)
            self.save_dirs[t] = (live, spoof)
            file_path = os.path.join(base_dir, "results.csv")
            self.results_files[t] = file_path
            with open(file_path, 'w', newline='') as f:
                csv.writer(f).writerow(['image_path', 'prediction', 'score', 'timestamp'])
            print(f"Threshold {t}: {base_dir}")
        self.debug = True

    def _log_result(self, t, path, pred, score):
        with open(self.results_files[t], 'a', newline='') as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv.writer(f).writerow([path, 'live' if pred==1 else 'spoof', f"{score:.4f}", ts])

    def process_frame(self, frame):
        result = LivenessDetector.predict_frame(frame, self.model, self.transform, self.device)
        if result is None:
            if self.debug:
                print("No face")
            return None
        _, score, mask, face_rgb, bbox, disp = result
        if face_rgb is None or bbox is None:
            if self.debug:
                print("Invalid face")
            return None
        if self.debug:
            print(f"Score: {score:.2f}")
        thresh_results = {}
        for t in self.thresholds:
            pred = 1 if score > t else 0
            live, spoof = self.save_dirs[t]
            cnt = self.frame_counters[t]
            fname = f"frame{cnt:03d}.jpg"
            save_dir = live if pred==1 else spoof
            path = os.path.join(save_dir, fname)
            cv2.imwrite(path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
            self._log_result(t, path, pred, score)
            self.frame_counters[t] += 1
            thresh_results[t] = {'label': pred, 'score': score, 'path': path}
        return {'bbox': bbox, 'display_face': disp, 'threshold_results': thresh_results, 'mask': mask}

class LivenessDetector:
    def __init__(self, checkpoint_path, thresholds):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.model = self.load_model(checkpoint_path)
        self.thresholds = thresholds

    @staticmethod
    def detect_and_crop_face(img, return_bbox=False, for_display=False):
        try:
            faces = DeepFace.extract_faces(img, detector_backend='ssd', enforce_detection=True)
            if not faces:
                return None if not return_bbox else (None, None)
            faces = sorted(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'], reverse=True)
            face = faces[0]['face']
            bbox = faces[0]['facial_area']
            if face is None or face.shape[0] == 0 or face.shape[1] == 0 or bbox['w'] < 50 or bbox['h'] < 50:
                return None if not return_bbox else (None, None)
            if face.max() <= 1:
                face = (face * 255).astype(np.uint8)
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            if len(face.shape)==3 and face.shape[2]==3 and not for_display:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            return (face, (bbox['x'], bbox['y'], bbox['w'], bbox['h'])) if return_bbox else face
        except Exception as e:
            print(e)
            return None if not return_bbox else (None, None)

    @staticmethod
    def get_predict(mask, label, threshold=0.7, score_type='combined'):
        with torch.no_grad():
            if score_type=='pixel':
                score = torch.mean(mask, axis=(1,2,3))
            elif score_type=='binary':
                score = label
            else:
                score = (torch.mean(mask, axis=(1,2,3)) + label)/2
            preds = (score > threshold).to(torch.uint16).cpu()
            return preds, score

    @staticmethod
    def predict_frame(frame, model, transform, device, threshold=0.7):
        disp_result = LivenessDetector.detect_and_crop_face(frame, True, True)
        if disp_result is None or disp_result[0] is None:
            return None
        disp_face, _ = disp_result
        proc_result = LivenessDetector.detect_and_crop_face(frame, True)
        if proc_result is None or proc_result[0] is None:
            return None
        face, bbox = proc_result
        image = transform(Image.fromarray(face)).unsqueeze(0).to(device)
        with torch.no_grad():
            mask_pred, label_pred = model(image)
            preds, score_tensor = LivenessDetector.get_predict(mask_pred, label_pred, threshold)
            score_val = score_tensor.item()
        return preds.item(), score_val, mask_pred.squeeze(), face, bbox, disp_face

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        if any("scale" in k for k in state.keys()):
            model_fp32 = PixelWise(pretrained=False).to(self.device)
            model_fp32.eval()
            model = torch.quantization.convert(model_fp32, inplace=False)
        else:
            model = PixelWise(pretrained=True).to(self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"Model loaded from {path}")
        return model

if __name__ == "__main__":
    checkpoint = "best_model2.pth"
    # checkpoint = "quantized_model.pth"
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    output_folder = "output/output_v3"
    detector = LivenessDetector(checkpoint, thresholds)
    manager = MultiThresholdManager(detector.model, detector.transform, detector.device, thresholds, output_folder)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit("Camera not accessible.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = manager.process_frame(frame)
        if results:
            bbox = results['bbox']
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
            # Display threshold results (bounding box, threshold, score, etc.)
            y_off = 30
            for t, res in sorted(results['threshold_results'].items()):
                label = "Live" if res['label'] == 1 else "Spoof"
                color = (0,255,0) if res['label'] == 1 else (0,0,255)
                text = f"T={t:.2f}: {label} ({res['score']:.2f})"
                cv2.putText(frame, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_off += 30
        cv2.imshow("Multi-Threshold Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

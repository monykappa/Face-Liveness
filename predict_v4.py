import cv2
import torch
from torchvision import transforms
from PIL import Image
from models.liveness_vit import LivenessViT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    image = Image.fromarray(frame).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image.to(device)

model = LivenessViT()
checkpoint_path = "best_model2.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
model.load_state_dict(model_state_dict, strict=False)
model.to(device)

def infer(frame, model):
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_frame(frame)
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
        label = "Live" if prediction > 0.5 else "Spoof"
        confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, confidence = infer(frame, model)

    # Display the result on the frame
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Liveness Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

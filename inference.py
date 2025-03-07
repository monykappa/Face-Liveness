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

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image.to(device)

model = LivenessViT()
checkpoint_path = "best_model3.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
print("start")
model.load_state_dict(model_state_dict, strict=False)
print("end")
model.to(device)
print("load success")

# model.eval()
# model.fuse_model()

# # Quantize the model
# model.qconfig = torch.ao.quantization.QConfig(
#     activation=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
#     weight=torch.ao.quantization.default_per_channel_weight_observer
# )
# torch.ao.quantization.prepare(model, inplace=True)
# torch.ao.quantization.convert(model, inplace=True)

def infer(image_path, model):
    model.eval()
    with torch.no_grad():
        input_tensor = load_and_preprocess_image(image_path)
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
        label = "Live" if prediction > 0.5 else "Spoof"
        confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

image_path = "images/image2.png"
label, confidence = infer(image_path, model)
print(f"Prediction: {label}, Confidence: {confidence:.4f}")
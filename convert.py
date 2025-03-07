import torch
from models.model import PixelWise

model = PixelWise(pretrained=False)  # disable auto-loading weights
ckpt = torch.load("best_model3.pth", map_location="cpu")
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load ignoring quantization keys; adjust model if needed.
model.load_state_dict(state_dict, strict=False)
model.eval()

example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("best_model5.pt")
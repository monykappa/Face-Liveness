import torch
import torch.nn as nn
import timm
import torch.quantization as quant
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

# class LivenessViT(nn.Module):
#     def __init__(self):
#         super(LivenessViT, self).__init__()
#         self.vit = timm.create_model("vit_basex _patch16_224", pretrained=False, num_classes=0)
#         self.feature_dim = 768
#         self.head = nn.Sequential(
#             nn.Linear(self.feature_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 1)
#         )
#         self.quant = quant.QuantStub()
#         self.dequant = quant.DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x)
#         features = self.vit(x)
#         out = self.head(features)
#         out = self.dequant(out)
#         return out

#     def fuse_model(self):
#         torch.ao.quantization.fuse_modules(self.head, [['0', '1']], inplace=True)

#     def get_features(self, x):
#         x = self.quant(x)
#         features = self.vit.forward_features(x)
#         return features

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self.hook_handles = []
#         self._register_hooks()

#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output.detach()

#         def backward_hook(module, grad_in, grad_out):
#             self.gradients = grad_out[0].detach()

#         self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
#         self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

#     def __call__(self, x, index=None):
#         self.model.eval()
#         features = self.model.get_features(x)
#         output = self.model.head(features.mean(dim=1))

#         if index is None:
#             index = torch.sigmoid(output) > 0.5

#         self.model.zero_grad()
#         output.backward(gradient=torch.ones_like(output))

#         gradients = self.gradients
#         activations = self.activations
#         pooled_gradients = torch.mean(gradients, dim=[0, 2])
#         for i in range(activations.size(1)):
#             activations[:, i, :] *= pooled_gradients[i]

#         heatmap = torch.mean(activations, dim=1).squeeze().cpu()
#         heatmap = F.relu(heatmap)
#         heatmap /= torch.max(heatmap) + 1e-8
#         return heatmap.numpy()

#     def cleanup(self):
#         for handle in self.hook_handles:
#             handle.remove()

# def superimpose_heatmap(heatmap, img, output_path):
#     img = img.squeeze().permute(1, 2, 0).cpu().numpy()
#     img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = heatmap * 0.4 + img * 255 * 0.6
#     cv2.imwrite(output_path, superimposed_img)


class LivenessViT(nn.Module):
    def __init__(self):
        super(LivenessViT, self).__init__()
        self.vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
        self.feature_dim = 192
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        features = self.vit(x)
        out = self.head(features)
        out = self.dequant(out)
        return out

    def fuse_model(self):
        torch.ao.quantization.fuse_modules(self.head, [['0', '1']], inplace=True)

    def get_features(self, x):
        x = self.quant(x)
        features = self.vit.forward_features(x)
        return features
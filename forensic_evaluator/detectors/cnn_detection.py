import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from urllib.request import urlretrieve
from .base import BaseDetector

class CNNDetectionDetector(BaseDetector):
    name = "CNNDetection"

    def load(self, model_path=None, device='cuda'):
        if model_path is None:
            model_path = 'weights/blur_jpg_prob0.5.pth'
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"
            urlretrieve(url, model_path)

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'fc.weight' in checkpoint:
            state_dict = checkpoint
        else:
            raise ValueError("Unknown checkpoint format")

        fc_shape = state_dict['fc.weight'].shape[0]
        self.model = models.resnet50(num_classes=fc_shape)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device).eval()
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def predict(self, image: Image.Image) -> dict:
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(img_t)
        # Single output, positive = fake
        if logit.shape[1] == 1:
            confidence = torch.sigmoid(logit[:,0]).item()
            logits = torch.cat([-logit, logit], dim=1)
        else:
            confidence = torch.softmax(logit, dim=1)[:,1].item()
            logits = logit
        return {'confidence': confidence, 'logits': logits.cpu().numpy().flatten(), 'features': None}

    def to(self, device):
        self.model.to(device)
        self.device = device

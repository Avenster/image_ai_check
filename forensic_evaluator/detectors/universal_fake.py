import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import clip
from .base import BaseDetector
import wget
import os

class UniversalFakeDetector(BaseDetector):
    name = "UniversalFakeDetect"

    def load(self, model_path=None, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        # Load fine-tuned weights from UniversalFakeDetect (if available)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=False)
        else:
            print("[UniversalFakeDetect] Using zero-shot CLIP weights; performance may be lower.")
        self.model.eval()
        # Preprocessing: CLIP requires 224x224 center-crop
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def predict(self, image: Image.Image) -> dict:
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(["a real photograph", "a fake generated image"]).to(self.device)
        with torch.no_grad():
            logits_per_image, _ = self.model(img_t, text)
            probs = torch.softmax(logits_per_image, dim=1)
        confidence = probs[0,1].item()
        return {'confidence': confidence, 'logits': logits_per_image.cpu().numpy().flatten(), 'features': None}

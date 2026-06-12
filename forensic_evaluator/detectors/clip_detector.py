import torch
from PIL import Image
import clip
from .base import BaseDetector

class CLIPZeroShotDetector(BaseDetector):
    name = "CLIP-Zero-Shot"

    def load(self, model_name="ViT-L/14", device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.text_inputs = clip.tokenize(["a real photograph", "a fake generated image"]).to(device)

    def predict(self, image: Image.Image) -> dict:
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_per_image, _ = self.model(img_t, self.text_inputs)
            probs = torch.softmax(logits_per_image, dim=1)
        confidence = probs[0,1].item()
        return {'confidence': confidence, 'logits': logits_per_image.cpu().numpy().flatten(), 'features': None}

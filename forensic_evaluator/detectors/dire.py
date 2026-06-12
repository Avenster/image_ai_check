import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import pipeline as hf_pipeline
import lpips
import os
from .base import BaseDetector
import wget

class DIREDetector(BaseDetector):
    name = "DIRE"

    def load(self, sd_model_id="runwayml/stable-diffusion-v1-5", device='cuda'):
        self.device = device
        self.sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model_id,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            safety_checker=None
        ).to(device)
        self.sd_pipe.enable_attention_slicing()
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        # Load a simple logistic regression to convert DIRE score to confidence
        # We'll use a fixed threshold for simplicity; in practice train a small model.
        # Here we approximate: higher reconstruction error = more real.
        # We'll normalize error by dividing by a typical value.
        self.threshold = 0.1  # default threshold for binary decision

    def predict(self, image: Image.Image) -> dict:
        w, h = image.size
        # Resize to multiples of 8
        w8 = (w // 8) * 8
        h8 = (h // 8) * 8
        img_resized = image.resize((w8, h8), Image.LANCZOS)
        img_tensor = torch.from_numpy(np.array(img_resized).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(self.device)
        # Add noise and denoise (strength=0.1)
        with torch.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
            generated = self.sd_pipe(
                prompt="", image=img_resized, strength=0.1,
                guidance_scale=1.0, output_type="pt"
            ).images
        gen_tensor = generated.squeeze(0)
        # Compute perceptual error (LPIPS)
        dist = self.loss_fn(img_tensor, gen_tensor).item()
        # Map error to confidence: lower error -> more fake. We scale to 0-1.
        # Using a simple logistic with calibrated parameters.
        fake_prob = 1.0 / (1.0 + np.exp(10*(dist - 0.1)))  # steepness 10, center 0.1
        return {'confidence': fake_prob, 'logits': None, 'features': dist}

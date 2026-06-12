from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

class RobustnessTester:
    @staticmethod
    def apply_perturbations(image: Image.Image):
        perturbations = {}
        # Resize
        for scale in [0.5, 0.75, 1.5]:
            w, h = image.size
            img_resized = image.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            img_back = img_resized.resize((w, h), Image.BILINEAR)
            perturbations[f'resize_{scale}'] = img_back
        # JPEG compression
        for q in [90, 70]:
            from io import BytesIO
            buf = BytesIO()
            image.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            perturbations[f'jpeg_{q}'] = Image.open(buf).convert('RGB')
        # Gaussian blur
        perturbations['blur_1'] = image.filter(ImageFilter.GaussianBlur(1))
        # Sharpening
        enhancer = ImageEnhance.Sharpness(image)
        perturbations['sharpen_2'] = enhancer.enhance(2.0)
        # Brightness/contrast
        enhancer = ImageEnhance.Brightness(image)
        perturbations['bright_1.1'] = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Contrast(image)
        perturbations['contrast_0.9'] = enhancer.enhance(0.9)
        return perturbations

import numpy as np
import cv2
from PIL import Image
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from .base import BaseDetector

class FrequencyDetector(BaseDetector):
    name = "Frequency-Domain"

    def load(self, model_path='weights/frequency_svm.pkl', device='cpu'):
        self.device = device
        if not os.path.exists(model_path):
            # Download a pre-trained SVM (placeholder URL – you can host one)
            # For now, create a dummy model that returns 0.5 (neutral) and print warning.
            print("[Frequency] No pre-trained SVM found. Using dummy model (always 0.5).")
            self.model = None
        else:
            self.model = joblib.load(model_path)
            self.scaler = StandardScaler()  # loaded from file in real case
            if os.path.exists(model_path.replace('.pkl', '_scaler.pkl')):
                self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        img_np = np.array(image.convert('RGB'))
        # Compute DCT of each 8x8 block and gather histogram
        features = []
        for ch in range(3):
            channel = img_np[:,:,ch].astype(np.float32)
            h, w = channel.shape
            dct_hist = []
            for i in range(0, h-7, 8):
                for j in range(0, w-7, 8):
                    block = channel[i:i+8, j:j+8]
                    dct_block = cv2.dct(block)
                    # take AC coefficients (skip DC)
                    dct_hist.extend(dct_block.ravel()[1:])
            dct_hist = np.array(dct_hist, dtype=np.float32)
            if dct_hist.size == 0 or np.std(dct_hist) < 1e-8:
                features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.append(float(np.mean(dct_hist)))
                features.append(float(np.std(dct_hist)))
                features.append(float(skew(dct_hist)))
                features.append(float(kurtosis(dct_hist)))
        return np.array(features).reshape(1, -1)

    def predict(self, image: Image.Image) -> dict:
        feat = self._extract_features(image)
        if self.model is None:
            return {'confidence': 0.5, 'logits': None, 'features': feat.flatten()}
        feat_scaled = self.scaler.transform(feat)
        proba = self.model.predict_proba(feat_scaled)[0,1]  # probability of fake
        return {'confidence': proba, 'logits': None, 'features': feat.flatten()}

from abc import ABC, abstractmethod
from PIL import Image
import torch

class BaseDetector(ABC):
    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> dict:
        # returns {confidence: float 0-1, logits: tensor or None, features: tensor or None}
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def to(self, device):
        pass

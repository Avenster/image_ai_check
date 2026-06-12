import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Dict

class EnsembleAggregator:
    def __init__(self, method='weighted_average', weights=None):
        self.method = method
        self.weights = weights

    def aggregate(self, detector_outputs: List[Dict], calibration_funcs=None) -> float:
        confidences = [d['confidence'] for d in detector_outputs if d.get('confidence') is not None]
        if not confidences:
            raise ValueError("No valid detector confidences to aggregate.")
        if calibration_funcs:
            confidences = [calib(c) for calib, c in zip(calibration_funcs, confidences)]

        if self.method == 'average':
            return np.mean(confidences)
        elif self.method == 'weighted_average':
            weights = self.weights if self.weights else np.ones(len(confidences))/len(confidences)
            if len(weights) != len(confidences):
                weights = np.ones(len(confidences)) / len(confidences)
            else:
                weights = np.array(weights, dtype=float)
                weights = weights / weights.sum()
            return np.dot(weights, confidences)
        elif self.method == 'voting':
            votes = [1 if c > 0.5 else 0 for c in confidences]
            return np.mean(votes)
        elif self.method == 'stacking':
            # Not implemented in real-time; requires pre-trained meta-classifier
            raise NotImplementedError("Stacking requires offline training.")
        else:
            raise ValueError(f"Unknown method: {self.method}")

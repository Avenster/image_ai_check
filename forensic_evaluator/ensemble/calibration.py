from sklearn.isotonic import IsotonicRegression
import numpy as np

class Calibrator:
    def __init__(self):
        self.models = []

    def fit(self, scores_list, labels):
        # scores_list: list of arrays (one per detector)
        # labels: true binary labels
        self.models = []
        for scores in scores_list:
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(scores, labels)
            self.models.append(ir)

    def calibrate(self, detector_idx, confidence):
        if self.models:
            return self.models[detector_idx].predict([confidence])[0]
        return confidence

import json
import numpy as np

def generate_report(image_path, detector_outputs, ensemble_score, disagreement_info, metrics=None):
    report = {
        "image": image_path,
        "ensemble_confidence": ensemble_score,
        "disagreement_analysis": disagreement_info,
        "detector_details": detector_outputs,
        "metrics": metrics
    }
    return json.dumps(report, indent=2, default=lambda x: str(x))

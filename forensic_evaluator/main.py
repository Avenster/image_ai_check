import argparse
import importlib
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
from ensemble.aggregator import EnsembleAggregator
from ensemble.calibration import Calibrator
from ensemble.disagreement import compute_disagreement
from evaluation.dataset import Dataset
from evaluation.metrics import compute_metrics
from evaluation.robustness import RobustnessTester
from evaluation.reporter import generate_report
from utils import load_config, get_device

DETECTOR_CLASSES = {
    'cnn_detection': ('detectors.cnn_detection', 'CNNDetectionDetector'),
    'dire': ('detectors.dire', 'DIREDetector'),
    'universal_fake': ('detectors.universal_fake', 'UniversalFakeDetector'),
    'clip_zero_shot': ('detectors.clip_detector', 'CLIPZeroShotDetector'),
    'frequency': ('detectors.frequency', 'FrequencyDetector'),
}

def build_detector(name):
    module_name, class_name = DETECTOR_CLASSES[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()

def main():
    parser = argparse.ArgumentParser()
    default_config = Path(__file__).resolve().parent / 'configs' / 'default.yaml'
    parser.add_argument('--config', default=str(default_config))
    parser.add_argument('--input', type=str, help='Single image path')
    parser.add_argument('--dataset', type=str, help='Dataset root folder (real/ and fake/ subfolders)')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--robustness', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    # Initialize detectors
    detectors = []
    # Order must match ensemble weights
    for name in DETECTOR_CLASSES:
        if config['detectors'][name]['enabled']:
            det_config = {k:v for k,v in config['detectors'][name].items() if k not in ['enabled']}
            if 'device' in det_config:
                det_config['device'] = get_device(det_config['device'])
            try:
                det = build_detector(name)
                det.load(**det_config)
                detectors.append(det)
                print(f"Loaded detector: {det.name}")
            except Exception as exc:
                print(f"[WARN] Skipping detector {name}: {exc}")

    if not detectors:
        raise RuntimeError("No detectors loaded. Install dependencies/models or disable unavailable detectors in config.")

    aggregator = EnsembleAggregator(
        method=config['ensemble']['method'],
        weights=config['ensemble']['weights'] if config['ensemble']['method']=='weighted_average' else None
    )

    # Single image mode
    if args.input:
        img = Image.open(args.input).convert('RGB')
        outputs = []
        for det in detectors:
            out = det.predict(img)
            outputs.append(out)
        # Calibration (not trained online; for demo we skip)
        ensemble_conf = aggregator.aggregate(outputs)
        ensemble_score = ensemble_conf * 100
        disagreement = compute_disagreement([o['confidence'] for o in outputs])

        report = generate_report(args.input, outputs, ensemble_score, disagreement)
        print(report)
        with open(os.path.join(args.output, 'report.json'), 'w') as f:
            f.write(report)
        print(f"Final ensemble confidence: {ensemble_score:.1f}% fake")

        if args.robustness:
            print("\nRobustness evaluation...")
            rb = RobustnessTester()
            perturbs = rb.apply_perturbations(img)
            for name, pimg in perturbs.items():
                pert_outputs = [det.predict(pimg) for det in detectors]
                pert_ensemble = aggregator.aggregate(pert_outputs)*100
                print(f"{name}: ensemble confidence = {pert_ensemble:.1f}%")

    # Batch mode
    elif args.dataset:
        dataset = Dataset(args.dataset)
        all_y_true = []
        all_y_scores = []
        for img, label in dataset:
            outputs = [det.predict(img) for det in detectors]
            conf = aggregator.aggregate(outputs)
            all_y_true.append(label)
            all_y_scores.append(conf)
        metrics = compute_metrics(all_y_true, all_y_scores)
        print("Batch Evaluation Metrics:")
        for k,v in metrics.items():
            print(f"{k}: {v}")
        with open(os.path.join(args.output, 'batch_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()

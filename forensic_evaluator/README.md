# Forensic Evaluator

Modular AI-image forensic detector evaluation framework.

## Install

```bash
cd forensic_evaluator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Single Image

```bash
python main.py --input /path/to/image.jpg --output results
```

The result is printed to the terminal and saved to `results/report.json`.

## Batch Dataset

Use this folder layout:

```text
dataset/
  real/
    image1.jpg
  fake/
    generator_name/
      image2.png
```

Run:

```bash
python main.py --dataset /path/to/dataset --output results
```

## Robustness Check

```bash
python main.py --input /path/to/image.jpg --output results --robustness
```

## Practical Notes

- The default config enables all detectors, but unavailable dependencies or missing model files are skipped with warnings.
- `Frequency-Domain` runs as a neutral dummy detector until `weights/frequency_svm.pkl` is trained/provided.
- `DIRE` needs substantial disk/GPU resources because it loads Stable Diffusion.
- For a quick smoke test, disable heavy detectors in `configs/default.yaml` and leave only `frequency` enabled.

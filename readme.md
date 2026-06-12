# Image AI Check

Research-oriented framework design for AI-image forensic detector evaluation, benchmarking, calibration, and reporting.

Start with the architecture document:

- `docs/forensic-evaluation-framework.md`

The project scope is detector evaluation and forensic analysis. It intentionally excludes detector-evasion, image-cleaning, metadata spoofing, or bypass workflows.

## Controlled Cleaner

The safe cleaner creates fixed, logged benchmark variants for robustness evaluation:

```bash
PYTHONPATH=src python -m forensic_eval.cleaner \
  --input path/to/image_or_dataset \
  --output-dir runs/cleaner_variants \
  --profile robustness
```

Outputs are written beside `variant_manifest.jsonl` with hashes, operations, and parameters.

# Business Entity Resolution Pipeline

High-performance, multi-stage Entity Resolution (ER) solution for the Amazon ML Challenge 2026.

## Structure
```
code/business_entity_resolution/
├── src/
│   ├── data_loader.py          # Robust TSV loading, literal "null" handling, dtype=str
│   ├── normalization.py        # Name & address normalization, legal entity stripping, tokenizers
│   ├── blocking.py             # Multi-strategy inverted indices (name, digit, word) with frequency capping
│   ├── candidate_generation.py # Partitioned candidate pair generation
│   ├── features.py             # Pairwise similarity feature extraction
│   ├── train.py                # Stratified entity split, hard negative mining, model training
│   ├── evaluate.py             # Exact macro F0.5 evaluation
│   ├── threshold.py            # Validation threshold search
│   ├── inference.py            # Test set batch inference
│   ├── output.py               # Formatted TSV export
│   └── main.py                 # CLI interface
├── README.md
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
python run_phase2_3.py
```

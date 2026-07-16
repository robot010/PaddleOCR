# Dataset Download Instructions

## 1. RxHandBD (Mendeley Data)

Download from: https://data.mendeley.com/datasets/dsb5r6vskg/1

```bash
# After download, extract to:
# data/rxhandbd/
#   images/
#   labels.csv
```

## 2. Doctor's Prescription BD (Kaggle)

Download from: https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset

```bash
# Requires Kaggle CLI:
kaggle datasets download -d mamun1113/doctors-handwritten-prescription-bd-dataset
unzip doctors-handwritten-prescription-bd-dataset.zip -d data/prescription_bd/
```

## 3. MedOCR-Vision (HuggingFace)

Download from: https://huggingface.co/datasets/naazimsnh02/medocr-vision-dataset

```python
from datasets import load_dataset
ds = load_dataset("naazimsnh02/medocr-vision-dataset")
# Export to data/medocr_vision/
```

## 4. Illegible Medical Prescription Images (Kaggle)

Download from: https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset

```bash
kaggle datasets download -d mehaksingal/illegible-medical-prescription-images-dataset
unzip illegible-medical-prescription-images-dataset.zip -d data/illegible_prescriptions/
```

## Directory Structure After Download

```
data/
├── rxhandbd/
│   ├── images/
│   └── labels.csv
├── prescription_bd/
│   ├── images/
│   └── annotations.csv
├── medocr_vision/
│   ├── images/
│   └── data.jsonl
└── illegible_prescriptions/
    └── ...
```

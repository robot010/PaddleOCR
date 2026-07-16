# Medical OCR Benchmark: PaddleOCR for Handwritten Prescription Recognition

Benchmarking PaddleOCR (PP-OCRv5, PaddleOCR-VL) against established OCR methods
on handwritten medical prescription datasets.

## Quick Start

```bash
# Install dependencies
pip install paddleocr pytesseract easyocr jiwer editdistance Pillow matplotlib pandas

# Download datasets (see data/README.md)

# Run benchmark
python run_benchmark.py --datasets rxhandbd --engines tesseract easyocr ppocr_v5 --n_runs 10

# Generate report
python generate_report.py --results_dir results/
```

## Project Structure

```
benchmark/medical_ocr/
├── data/                  # Dataset storage (gitignored)
├── results/               # Raw results (gitignored)
├── figures/               # Generated plots
├── engines.py             # OCR engine wrappers
├── datasets.py            # Dataset loaders
├── metrics.py             # CER, WER, Exact Match, F1
├── run_benchmark.py       # Main benchmark runner
└── generate_report.py     # Visualization and tables
```

## Datasets

1. **RxHandBD** — 5,578 handwritten prescription word images (Mendeley)
2. **Doctor's Prescription BD** — ~540 full prescriptions (Kaggle)
3. **MedOCR-Vision** — 1,462 medical samples (HuggingFace)

## OCR Engines

| Engine | Model Size | Type |
|--------|-----------|------|
| Tesseract v5 | ~15MB | Traditional + LSTM |
| EasyOCR | ~100MB | CRAFT + CRNN |
| PP-OCRv5 | ~5MB | Det + Rec pipeline |
| PaddleOCR-VL-1.5 | ~900MB | Vision-Language Model |

"""Dataset loaders for medical OCR benchmarking.

Each loader returns a list of (image_path, ground_truth_text) tuples.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple


class OCRSample(NamedTuple):
    """A single OCR evaluation sample."""

    image_path: Path
    ground_truth: str


class OCRDataset:
    """Container for an OCR evaluation dataset."""

    def __init__(self, name: str, samples: list[OCRSample]) -> None:
        self.name = name
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"OCRDataset(name='{self.name}', n_samples={len(self)})"

    @property
    def image_paths(self) -> list[Path]:
        return [s.image_path for s in self.samples]

    @property
    def ground_truths(self) -> list[str]:
        return [s.ground_truth for s in self.samples]


def load_rxhandbd(data_dir: str | Path) -> OCRDataset:
    """Load the RxHandBD dataset.

    Expected structure:
        data_dir/
            images/
                word_0001.jpg
                word_0002.jpg
                ...
            labels.csv   (or labels.txt with "filename,text" format)

    The dataset from Mendeley contains cropped word images with labels.
    Adapt the loader based on the actual file structure after download.
    """
    data_dir = Path(data_dir)
    samples = []

    # Try CSV format first
    label_file = data_dir / "labels.csv"
    if not label_file.exists():
        label_file = data_dir / "labels.txt"

    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) < 2:
                    continue
                img_name, text = row[0].strip(), row[1].strip()
                # Skip header row
                if row_idx == 0 and img_name.lower() in ("images", "image", "filename", "file"):
                    continue
                img_path = data_dir / "images" / img_name
                if not img_path.exists():
                    img_path = data_dir / img_name
                if img_path.exists() and text:
                    samples.append(OCRSample(img_path, text))
    else:
        # Fallback: try to find a directory structure where folder names are labels
        images_dir = data_dir / "images"
        if not images_dir.exists():
            images_dir = data_dir
        for class_dir in sorted(images_dir.iterdir()):
            if class_dir.is_dir():
                label = class_dir.name
                for img_file in sorted(class_dir.glob("*.jpg")):
                    samples.append(OCRSample(img_file, label))
                for img_file in sorted(class_dir.glob("*.png")):
                    samples.append(OCRSample(img_file, label))

    return OCRDataset("RxHandBD", samples)


def load_medocr_vision(data_dir: str | Path) -> OCRDataset:
    """Load the MedOCR-Vision dataset from HuggingFace.

    Expected structure (after `datasets` download or manual export):
        data_dir/
            data.jsonl  (or metadata.jsonl)
            images/
                0001.jpg
                ...

    Each JSONL line: {"image": "path", "text": "ground truth"}
    """
    data_dir = Path(data_dir)
    samples = []

    for jsonl_name in ["data.jsonl", "metadata.jsonl", "train.jsonl"]:
        jsonl_path = data_dir / jsonl_name
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    img_name = entry.get("image", entry.get("file_name", ""))
                    text = entry.get("text", entry.get("ground_truth", ""))
                    img_path = data_dir / img_name
                    if not img_path.exists():
                        img_path = data_dir / "images" / img_name
                    if img_path.exists() and text:
                        samples.append(OCRSample(img_path, text))
            break

    return OCRDataset("MedOCR-Vision", samples)


def load_prescription_bd(data_dir: str | Path) -> OCRDataset:
    """Load the Doctor's Prescription BD dataset from Kaggle.

    This dataset contains full-page prescription images.
    Ground truth format varies — adapt based on actual download.

    Expected structure:
        data_dir/
            images/
                prescription_001.jpg
                ...
            annotations.json  (or .csv)
    """
    data_dir = Path(data_dir)
    samples = []

    # Try JSON annotations
    for ann_name in ["annotations.json", "labels.json"]:
        ann_path = data_dir / ann_name
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            if isinstance(annotations, list):
                for entry in annotations:
                    img_name = entry.get("image", entry.get("file_name", ""))
                    text = entry.get("text", entry.get("ground_truth", ""))
                    img_path = data_dir / img_name
                    if not img_path.exists():
                        img_path = data_dir / "images" / img_name
                    if img_path.exists() and text:
                        samples.append(OCRSample(img_path, text))
            break

    # Try CSV annotations
    if not samples:
        for csv_name in ["annotations.csv", "labels.csv"]:
            csv_path = data_dir / csv_name
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_name = row.get("image", row.get("file_name", ""))
                        text = row.get("text", row.get("ground_truth", ""))
                        img_path = data_dir / img_name
                        if not img_path.exists():
                            img_path = data_dir / "images" / img_name
                        if img_path.exists() and text:
                            samples.append(OCRSample(img_path, text))
                break

    return OCRDataset("PrescriptionBD", samples)


def load_dataset(name: str, data_dir: str | Path) -> OCRDataset:
    """Load a dataset by name.

    Args:
        name: One of 'rxhandbd', 'medocr_vision', 'prescription_bd'.
        data_dir: Path to the dataset root directory.

    Returns:
        An OCRDataset instance.
    """
    loaders = {
        "rxhandbd": load_rxhandbd,
        "medocr_vision": load_medocr_vision,
        "prescription_bd": load_prescription_bd,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(loaders.keys())}")
    return loaders[name](data_dir)

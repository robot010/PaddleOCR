"""Qualitative evaluation on unlabeled prescription images.

Runs OCR engines on images without ground truth and produces
side-by-side comparison outputs for visual analysis.

Usage:
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python qualitative_eval.py \
        --image_dir data/illegible_prescriptions/images \
        --engines tesseract ppocr_v5 \
        --max_images 10 \
        --output_dir results/qualitative
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from engines import get_engine


def run_qualitative_eval(
    image_dir: str,
    engine_names: list[str],
    max_images: int = 10,
    output_dir: str = "results/qualitative",
) -> None:
    """Run OCR engines on unlabeled images and save comparison results."""
    image_dir = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: p.name,
    )
    if max_images > 0:
        image_paths = image_paths[:max_images]

    print(f"Evaluating {len(image_paths)} images with {len(engine_names)} engines\n")

    # Initialize engines
    engines = {}
    for name in engine_names:
        print(f"Loading {name}...")
        engines[name] = get_engine(name)

    # Run evaluation
    results = []
    for img_path in image_paths:
        entry = {"image": img_path.name, "outputs": {}}
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"{'='*60}")

        for eng_name, engine in engines.items():
            text, latency = engine.recognize_timed(img_path)
            word_count = len(text.split()) if text else 0
            entry["outputs"][eng_name] = {
                "text": text,
                "word_count": word_count,
                "latency_ms": round(latency, 1),
            }
            # Truncate display for readability
            display = text[:150] + "..." if len(text) > 150 else text
            print(f"\n  [{engine.name}] ({latency:.0f}ms, {word_count} words)")
            print(f"  {display}")

        results.append(entry)

    # Save results
    results_file = output_path / "qualitative_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Average words extracted per engine")
    print(f"{'='*60}")
    for eng_name in engine_names:
        word_counts = [r["outputs"][eng_name]["word_count"] for r in results]
        latencies = [r["outputs"][eng_name]["latency_ms"] for r in results]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        print(f"  {eng_name:<20} avg_words={avg_words:.1f}  avg_latency={avg_lat:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative OCR evaluation")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument(
        "--engines", nargs="+", required=True,
        choices=["tesseract", "easyocr", "ppocr_v5", "paddleocr_vl"],
    )
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/qualitative")
    args = parser.parse_args()

    run_qualitative_eval(
        image_dir=args.image_dir,
        engine_names=args.engines,
        max_images=args.max_images,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

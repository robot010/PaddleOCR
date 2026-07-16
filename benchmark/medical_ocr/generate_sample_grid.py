"""Generate side-by-side comparison grids from RxHandBD samples.

Runs OCR engines on selected images and renders annotated comparison grids
showing: original image, ground truth, and each engine's prediction.

Usage:
    # Run engines live on a random subset:
    python generate_sample_grid.py \
        --data_dir data/rxhandbd \
        --engines tesseract easyocr ppocr_v5 glm_ocr \
        --n_samples 20 \
        --output_dir figures/samples

    # Use cached predictions (skip engine loading):
    python generate_sample_grid.py \
        --data_dir data/rxhandbd \
        --cache figures/samples/predictions.json \
        --output_dir figures/samples
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from datasets import load_rxhandbd
from metrics import character_error_rate, word_error_rate, exact_match


def run_predictions(
    image_paths: list[Path],
    ground_truths: list[str],
    engine_names: list[str],
) -> list[dict]:
    """Run engines on images and return per-sample predictions."""
    from engines import get_engine

    engines = {}
    for name in engine_names:
        print(f"Loading {name}...", flush=True)
        engines[name] = get_engine(name)

    results = []
    for i, (img_path, gt) in enumerate(zip(image_paths, ground_truths)):
        print(f"  [{i + 1}/{len(image_paths)}] {img_path.name}", flush=True)
        entry = {
            "image": str(img_path),
            "image_name": img_path.name,
            "ground_truth": gt,
            "predictions": {},
        }
        for eng_name, engine in engines.items():
            text, latency = engine.recognize_timed(img_path)
            cer = character_error_rate(text, gt)
            wer = word_error_rate(text, gt)
            em = exact_match(text, gt)
            entry["predictions"][eng_name] = {
                "text": text,
                "latency_ms": round(latency, 1),
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "exact_match": em == 1.0,
            }
        results.append(entry)
    return results


def select_representative_samples(
    results: list[dict],
    engine_names: list[str],
) -> list[dict]:
    """Pick ~6 representative samples: easy, medium, hard, mixed winners, all-fail."""
    # Categorize by how many engines got exact match
    buckets: dict[int, list[dict]] = {}
    for r in results:
        n_correct = sum(
            1 for e in engine_names if r["predictions"].get(e, {}).get("exact_match", False)
        )
        buckets.setdefault(n_correct, []).append(r)

    selected = []
    selected_entries = set()

    def _already_selected(r: dict) -> bool:
        return id(r) in selected_entries

    def _add(label: str, r: dict) -> None:
        selected.append((label, r))
        selected_entries.add(id(r))

    # 1. Both modern engines correct, old engines wrong
    for r in results:
        preds = r["predictions"]
        old_wrong = all(
            not preds.get(e, {}).get("exact_match", False) for e in ["tesseract", "easyocr"]
        )
        new_both_right = all(
            preds.get(e, {}).get("exact_match", False) for e in ["ppocr_v5", "glm_ocr"]
        )
        if old_wrong and new_both_right and not _already_selected(r):
            _add("Both modern engines correct", r)
            break

    # 2. Only GLM-OCR correct (character advantage)
    for r in results:
        preds = r["predictions"]
        glm_right = preds.get("glm_ocr", {}).get("exact_match", False)
        pp_wrong = not preds.get("ppocr_v5", {}).get("exact_match", False)
        if glm_right and pp_wrong and not _already_selected(r):
            _add("GLM-OCR only", r)
            break

    # 3. Only PP-OCRv5 correct
    for r in results:
        preds = r["predictions"]
        pp_right = preds.get("ppocr_v5", {}).get("exact_match", False)
        glm_wrong = not preds.get("glm_ocr", {}).get("exact_match", False)
        if pp_right and glm_wrong and not _already_selected(r):
            _add("PP-OCRv5 only", r)
            break

    # 4. GLM-OCR has much lower CER than PP-OCRv5 (but neither exact)
    for r in results:
        preds = r["predictions"]
        glm_cer = preds.get("glm_ocr", {}).get("cer", 1.0)
        pp_cer = preds.get("ppocr_v5", {}).get("cer", 1.0)
        neither_exact = (
            not preds.get("glm_ocr", {}).get("exact_match", False)
            and not preds.get("ppocr_v5", {}).get("exact_match", False)
        )
        if neither_exact and glm_cer < pp_cer * 0.6 and not _already_selected(r):
            _add("GLM-OCR closer", r)
            break

    return selected


def render_comparison_grid(
    results: list[dict],
    engine_names: list[str],
    output_path: Path,
    title: str = "OCR Engine Comparison on RxHandBD Samples",
) -> None:
    """Render a grid comparing engine predictions on sample images."""
    n_samples = len(results)
    n_engines = len(engine_names)

    # Display-friendly engine names
    display_names = {
        "tesseract": "Tesseract",
        "easyocr": "EasyOCR",
        "ppocr_v5": "PP-OCRv5",
        "glm_ocr": "GLM-OCR",
    }

    fig, axes = plt.subplots(
        n_samples, 1,
        figsize=(14, 3.0 * n_samples),
        squeeze=False,
    )

    for row, (label, entry) in enumerate(results):
        ax = axes[row, 0]
        ax.set_axis_off()

        gt = entry["ground_truth"]
        img_path = entry["image"]

        # Try to load and show the image
        try:
            img = mpimg.imread(img_path)
            # Create an inset axes for the image on the left
            img_ax = ax.inset_axes([0.0, 0.1, 0.18, 0.8])
            img_ax.imshow(img)
            img_ax.set_axis_off()
        except Exception:
            pass

        # Build text block
        lines = []
        lines.append(f"Ground Truth:  {gt}")
        lines.append("")
        for eng in engine_names:
            pred_data = entry["predictions"].get(eng, {})
            pred_text = pred_data.get("text", "N/A")
            is_exact = pred_data.get("exact_match", False)
            cer_val = pred_data.get("cer", -1)

            marker = "EXACT" if is_exact else f"CER={cer_val:.2f}"
            color_marker = "green" if is_exact else "red"
            disp = display_names.get(eng, eng)
            lines.append(f"{disp:>12}:  {pred_text}  [{marker}]")

        text_block = "\n".join(lines)
        ax.text(
            0.22, 0.5, text_block,
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        # Row label
        ax.text(
            0.0, 1.02, f"({row + 1}) {label}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved grid: {output_path}")


def render_individual_cards(
    results: list[dict],
    engine_names: list[str],
    output_dir: Path,
) -> None:
    """Render individual comparison cards (one per sample) — better for blog embedding."""
    display_names = {
        "tesseract": "Tesseract",
        "easyocr": "EasyOCR",
        "ppocr_v5": "PP-OCRv5",
        "glm_ocr": "GLM-OCR",
    }

    for i, (label, entry) in enumerate(results):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 3]})

        # Left: image
        ax_img = axes[0]
        try:
            img = mpimg.imread(entry["image"])
            ax_img.imshow(img)
        except Exception:
            ax_img.text(0.5, 0.5, "Image\nnot found", ha="center", va="center")
        ax_img.set_axis_off()
        ax_img.set_title(f"Image: {entry['image_name']}", fontsize=9)

        # Right: predictions table
        ax_text = axes[1]
        ax_text.set_axis_off()

        gt = entry["ground_truth"]
        rows = [["Engine", "Prediction", "CER", "Match"]]
        cell_colors = [["lightgray"] * 4]

        for eng in engine_names:
            pred_data = entry["predictions"].get(eng, {})
            pred_text = pred_data.get("text", "N/A")
            is_exact = pred_data.get("exact_match", False)
            cer_val = pred_data.get("cer", -1)

            rows.append([
                display_names.get(eng, eng),
                pred_text[:40] + ("..." if len(pred_text) > 40 else ""),
                f"{cer_val:.3f}",
                "Yes" if is_exact else "No",
            ])
            row_color = "#d4edda" if is_exact else "#f8d7da"
            cell_colors.append([row_color] * 4)

        table = ax_text.table(
            cellText=rows,
            cellColours=cell_colors,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        fig.suptitle(
            f"({i + 1}) {label}  |  Ground Truth: \"{gt}\"",
            fontsize=11, fontweight="bold",
        )

        plt.tight_layout()
        card_path = output_dir / f"sample_{i + 1:02d}_{label.lower().replace(' ', '_')}.png"
        plt.savefig(card_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {card_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RxHandBD comparison grids")
    parser.add_argument("--data_dir", type=str, default="data/rxhandbd")
    parser.add_argument(
        "--engines", nargs="+",
        default=["tesseract", "easyocr", "ppocr_v5", "glm_ocr"],
    )
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of random samples to evaluate (before selection)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to cached predictions JSON (skip engine loading)")
    parser.add_argument("--output_dir", type=str, default="figures/samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache and Path(args.cache).exists():
        print(f"Loading cached predictions from {args.cache}")
        with open(args.cache, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        # Load dataset and sample
        dataset = load_rxhandbd(args.data_dir)
        print(f"Dataset: {dataset} ({len(dataset)} samples)")

        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))
        sample_paths = [dataset.samples[i].image_path for i in indices]
        sample_gts = [dataset.samples[i].ground_truth for i in indices]

        # Run predictions
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        all_results = run_predictions(sample_paths, sample_gts, args.engines)

        # Cache results
        cache_path = output_dir / "predictions.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Cached predictions to {cache_path}")

    # Select representative examples
    selected = select_representative_samples(all_results, args.engines)
    print(f"\nSelected {len(selected)} representative samples:")
    for label, entry in selected:
        print(f"  {label}: {entry['image_name']} (GT: {entry['ground_truth']})")

    # Render outputs
    render_comparison_grid(
        selected, args.engines,
        output_dir / "comparison_grid.png",
    )
    render_individual_cards(selected, args.engines, output_dir)

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()

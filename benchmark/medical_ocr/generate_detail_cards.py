"""Generate detail cards for the blog's side-by-side examples section.

Uses specific samples from predictions.json that differ from the comparison grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ENGINE_NAMES = ["tesseract", "easyocr", "ppocr_v5", "glm_ocr"]
DISPLAY_NAMES = {
    "tesseract": "Tesseract",
    "easyocr": "EasyOCR",
    "ppocr_v5": "PP-OCRv5",
    "glm_ocr": "GLM-OCR",
}

# Hand-picked samples (different from comparison grid which uses Ronem, Vineet, Eylox, Ambrox)
DETAIL_SAMPLES = [
    ("P3346.jpg", "GLM-OCR nails a complex drug name"),
    ("P0212.jpg", "PP-OCRv5 closer, GLM-OCR hallucinates"),
    ("P0597.jpg", "GLM-OCR nearly perfect"),
    ("P3929.jpg", "GLM-OCR outputs LaTeX"),
]


def render_card(
    entry: dict,
    label: str,
    index: int,
    output_dir: Path,
) -> Path:
    """Render a single comparison card."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 3]})

    ax_img = axes[0]
    try:
        img = mpimg.imread(entry["image"])
        ax_img.imshow(img)
    except Exception:
        ax_img.text(0.5, 0.5, "Image\nnot found", ha="center", va="center")
    ax_img.set_axis_off()
    ax_img.set_title(f"Image: {entry['image_name']}", fontsize=9)

    ax_text = axes[1]
    ax_text.set_axis_off()

    gt = entry["ground_truth"]
    rows = [["Engine", "Prediction", "CER", "Match"]]
    cell_colors = [["lightgray"] * 4]

    for eng in ENGINE_NAMES:
        pred_data = entry["predictions"].get(eng, {})
        pred_text = pred_data.get("text", "N/A")
        is_exact = pred_data.get("exact_match", False)
        cer_val = pred_data.get("cer", -1)

        rows.append([
            DISPLAY_NAMES.get(eng, eng),
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
        f"({index}) {label}  |  Ground Truth: \"{gt}\"",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_").replace(",", "")
    card_path = output_dir / f"detail_{index:02d}_{safe_label}.png"
    plt.savefig(card_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {card_path}")
    return card_path


def main() -> None:
    cache_path = Path("figures/samples/predictions.json")
    output_dir = Path("figures/samples")

    with open(cache_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # Index by image name
    by_name = {r["image_name"]: r for r in all_results}

    for i, (img_name, label) in enumerate(DETAIL_SAMPLES, start=1):
        if img_name not in by_name:
            print(f"WARNING: {img_name} not found in predictions cache, skipping")
            continue
        render_card(by_name[img_name], label, i, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()

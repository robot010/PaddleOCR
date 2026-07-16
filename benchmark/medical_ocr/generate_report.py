"""Generate visualizations and tables from benchmark results.

Usage:
    python generate_report.py --results_dir results/ --output_dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Display-friendly names for engines and datasets
ENGINE_DISPLAY = {
    "tesseract": "Tesseract",
    "easyocr": "EasyOCR",
    "ppocr_v5": "PP-OCRv5",
    "glm_ocr": "GLM-OCR",
}
DATASET_DISPLAY = {
    "rxhandbd": "RxHandBD",
}


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load all JSON result files into a single DataFrame."""
    results_dir = Path(results_dir)
    records = []

    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                if "error" not in entry:
                    records.append(entry)

    if not records:
        raise FileNotFoundError(f"No valid result files found in {results_dir}")

    df = pd.DataFrame(records)
    # Deduplicate if combined + per-pair files overlap
    if "run_idx" in df.columns:
        df = df.drop_duplicates(subset=["engine_key", "dataset", "run_idx"])
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table with mean +/- 95% CI per engine-dataset pair."""
    def agg_with_ci(series: pd.Series) -> str:
        mean = series.mean()
        if len(series) > 1:
            ci = 1.96 * series.std(ddof=1) / np.sqrt(len(series))
            return f"{mean:.4f} +/- {ci:.4f}"
        return f"{mean:.4f}"

    grouped = df.groupby(["dataset", "engine_key"])

    rows = []
    for (dataset, engine), group in grouped:
        rows.append(
            {
                "Dataset": dataset,
                "Engine": group["engine"].iloc[0] if "engine" in group.columns else engine,
                "CER": agg_with_ci(group["cer"]),
                "WER": agg_with_ci(group["wer"]),
                "Exact Match": agg_with_ci(group["exact_match_rate"]),
                "Latency (ms)": agg_with_ci(group["mean_latency_ms"]),
                "N Runs": len(group),
            }
        )

    return pd.DataFrame(rows)


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    lower_is_better: bool = True,
) -> None:
    """Create a grouped bar chart comparing engines across datasets for one metric."""
    datasets = sorted(df["dataset"].unique())
    engines = sorted(df["engine_key"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.8 / len(engines)

    for i, engine in enumerate(engines):
        means = []
        cis = []
        for dataset in datasets:
            subset = df[(df["engine_key"] == engine) & (df["dataset"] == dataset)]
            if len(subset) > 0:
                mean = subset[metric].mean()
                ci = 1.96 * subset[metric].std(ddof=1) / np.sqrt(len(subset)) if len(subset) > 1 else 0
                means.append(mean)
                cis.append(ci)
            else:
                means.append(0)
                cis.append(0)

        offset = (i - len(engines) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=cis, label=ENGINE_DISPLAY.get(engine, engine), capsize=3)

    display_labels = [DATASET_DISPLAY.get(d, d) for d in datasets]

    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0, ha="center")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_latency(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter plot: accuracy (1 - CER) vs mean latency per engine."""
    engines = sorted(df["engine_key"].unique())

    fig, ax = plt.subplots(figsize=(8, 6))

    for engine in engines:
        subset = df[df["engine_key"] == engine]
        accuracy = 1 - subset["cer"].mean()
        latency = subset["mean_latency_ms"].mean()
        disp = ENGINE_DISPLAY.get(engine, engine)
        ax.scatter(latency, accuracy, s=100, label=disp, zorder=5)
        ax.annotate(disp, (latency, accuracy), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Mean Latency (ms/image)")
    ax.set_ylabel("Accuracy (1 - CER)")
    ax.set_title("Accuracy vs Latency Trade-off")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all(results_dir: str, output_dir: str) -> None:
    """Generate all report artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    print(f"Loaded {len(df)} result records.")

    # Summary table
    summary = summary_table(df)
    print("\n" + summary.to_string(index=False))
    summary.to_csv(output_path / "summary_table.csv", index=False)
    print(f"\nSaved: {output_path / 'summary_table.csv'}")

    # Metric comparison plots
    plot_metric_comparison(
        df, "cer", "Character Error Rate (CER)", "CER Comparison by Engine and Dataset",
        output_path / "cer_comparison.png", lower_is_better=True,
    )
    plot_metric_comparison(
        df, "wer", "Word Error Rate (WER)", "WER Comparison by Engine and Dataset",
        output_path / "wer_comparison.png", lower_is_better=True,
    )
    plot_metric_comparison(
        df, "exact_match_rate", "Exact Match Rate", "Exact Match Comparison",
        output_path / "exact_match_comparison.png", lower_is_better=False,
    )
    plot_metric_comparison(
        df, "mean_latency_ms", "Latency (ms/image)", "Inference Latency Comparison",
        output_path / "latency_comparison.png", lower_is_better=True,
    )

    # Accuracy vs latency scatter
    plot_accuracy_vs_latency(df, output_path / "accuracy_vs_latency.png")

    # Export LaTeX table for paper
    _export_latex_table(summary, output_path / "summary_table.tex")


def _export_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Export summary as a LaTeX table for the paper."""
    latex = summary.to_latex(index=False, escape=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="figures",
        help="Directory for output figures and tables.",
    )
    args = parser.parse_args()
    generate_all(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()

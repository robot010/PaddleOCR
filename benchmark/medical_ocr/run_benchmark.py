"""Main benchmark runner for medical OCR evaluation.

Usage:
    python run_benchmark.py \
        --datasets rxhandbd \
        --data_dirs data/rxhandbd \
        --engines tesseract easyocr ppocr_v5 \
        --n_runs 10 \
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from datasets import load_dataset
from engines import get_engine
from metrics import compute_metrics, compute_per_sample_metrics


def run_single_evaluation(
    engine_name: str,
    dataset_name: str,
    data_dir: str,
    engine_kwargs: dict | None = None,
) -> dict:
    """Run a single evaluation of one engine on one dataset.

    Returns a dict with metrics, per-sample results, and timing info.
    """
    engine_kwargs = engine_kwargs or {}
    dataset = load_dataset(dataset_name, data_dir)

    if len(dataset) == 0:
        raise ValueError(f"Dataset '{dataset_name}' at '{data_dir}' is empty.")

    engine = get_engine(engine_name, **engine_kwargs)

    predictions = []
    latencies_ms = []

    for sample in dataset.samples:
        text, elapsed = engine.recognize_timed(sample.image_path)
        predictions.append(text)
        latencies_ms.append(elapsed)

    metrics = compute_metrics(predictions, dataset.ground_truths)
    per_sample = compute_per_sample_metrics(predictions, dataset.ground_truths)

    return {
        "engine": engine.name,
        "engine_key": engine_name,
        "dataset": dataset_name,
        "n_samples": len(dataset),
        "cer": metrics.cer,
        "wer": metrics.wer,
        "exact_match_rate": metrics.exact_match_rate,
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "total_time_s": sum(latencies_ms) / 1000,
        "per_sample": per_sample,
        "predictions": predictions,
        "references": dataset.ground_truths,
    }


def run_benchmark(
    datasets: list[str],
    data_dirs: list[str],
    engines: list[str],
    n_runs: int = 10,
    output_dir: str = "results",
    engine_kwargs: dict[str, dict] | None = None,
) -> list[dict]:
    """Run the full benchmark: multiple engines x datasets x runs.

    Args:
        datasets: List of dataset names.
        data_dirs: List of data directory paths (parallel to datasets).
        engines: List of engine names.
        n_runs: Number of independent runs per engine-dataset pair.
        output_dir: Directory to save results.
        engine_kwargs: Optional per-engine kwargs.

    Returns:
        List of result dicts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    engine_kwargs = engine_kwargs or {}

    all_results = []
    total_combos = len(engines) * len(datasets) * n_runs
    combo_idx = 0

    for engine_name in engines:
        for dataset_name, data_dir in zip(datasets, data_dirs):
            run_results = []
            for run_idx in range(n_runs):
                combo_idx += 1
                print(
                    f"[{combo_idx}/{total_combos}] "
                    f"{engine_name} x {dataset_name} (run {run_idx + 1}/{n_runs})"
                )
                try:
                    result = run_single_evaluation(
                        engine_name,
                        dataset_name,
                        data_dir,
                        engine_kwargs.get(engine_name, {}),
                    )
                    result["run_idx"] = run_idx
                    run_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    run_results.append(
                        {
                            "engine_key": engine_name,
                            "dataset": dataset_name,
                            "run_idx": run_idx,
                            "error": str(e),
                        }
                    )

            all_results.extend(run_results)

            # Save intermediate results per engine-dataset pair
            pair_file = output_path / f"{engine_name}_{dataset_name}.json"
            _save_results(run_results, pair_file)

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_path / f"benchmark_{timestamp}.json"
    _save_results(all_results, combined_file)

    # Print summary
    _print_summary(all_results, engines, datasets, n_runs)

    return all_results


def _save_results(results: list[dict], path: Path) -> None:
    """Save results to JSON, excluding large per-sample data for combined file."""
    # Strip per-sample data and raw predictions/references for compact saving
    compact = []
    for r in results:
        entry = {k: v for k, v in r.items() if k not in ("per_sample", "predictions", "references")}
        compact.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)


def _print_summary(
    results: list[dict],
    engines: list[str],
    datasets: list[str],
    n_runs: int,
) -> None:
    """Print a summary table of results with mean +/- 95% CI."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for dataset_name in datasets:
        print(f"\nDataset: {dataset_name}")
        print(f"{'Engine':<25} {'CER':>12} {'WER':>12} {'EM Rate':>12} {'Latency(ms)':>14}")
        print("-" * 75)

        for engine_name in engines:
            engine_results = [
                r
                for r in results
                if r.get("engine_key") == engine_name
                and r.get("dataset") == dataset_name
                and "error" not in r
            ]
            if not engine_results:
                print(f"{engine_name:<25} {'ERROR':>12}")
                continue

            cers = [r["cer"] for r in engine_results]
            wers = [r["wer"] for r in engine_results]
            ems = [r["exact_match_rate"] for r in engine_results]
            lats = [r["mean_latency_ms"] for r in engine_results]

            def fmt(values: list[float]) -> str:
                mean = np.mean(values)
                if len(values) > 1:
                    ci95 = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
                    return f"{mean:.4f}+/-{ci95:.4f}"
                return f"{mean:.4f}"

            print(
                f"{engine_name:<25} "
                f"{fmt(cers):>12} "
                f"{fmt(wers):>12} "
                f"{fmt(ems):>12} "
                f"{fmt(lats):>14}"
            )

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical OCR Benchmark Runner")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=["rxhandbd", "medocr_vision", "prescription_bd"],
        help="Datasets to evaluate on.",
    )
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        required=True,
        help="Paths to dataset directories (parallel to --datasets).",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        required=True,
        choices=["tesseract", "easyocr", "ppocr_v5", "paddleocr_vl", "doctr", "glm_ocr", "mineru"],
        help="OCR engines to evaluate.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of independent runs (default: 10).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/).",
    )
    args = parser.parse_args()

    if len(args.datasets) != len(args.data_dirs):
        parser.error("--datasets and --data_dirs must have the same number of arguments.")

    run_benchmark(
        datasets=args.datasets,
        data_dirs=args.data_dirs,
        engines=args.engines,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

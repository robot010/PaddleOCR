"""Metrics for OCR evaluation: CER, WER, Exact Match, and drug name F1."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import editdistance


def _normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, normalize unicode."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def character_error_rate(prediction: str, reference: str) -> float:
    """Character Error Rate (CER) = edit_distance(pred, ref) / len(ref).

    Returns 0.0 for empty reference and empty prediction.
    Returns 1.0 for empty reference with non-empty prediction.
    """
    pred = _normalize_text(prediction)
    ref = _normalize_text(reference)
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, ref) / len(ref)


def word_error_rate(prediction: str, reference: str) -> float:
    """Word Error Rate (WER) = edit_distance(pred_words, ref_words) / len(ref_words)."""
    pred_words = _normalize_text(prediction).split()
    ref_words = _normalize_text(reference).split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return editdistance.eval(pred_words, ref_words) / len(ref_words)


def exact_match(prediction: str, reference: str) -> float:
    """Exact match after normalization. Returns 1.0 if match, 0.0 otherwise."""
    return 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0


@dataclass
class OCRMetrics:
    """Aggregated OCR metrics for a set of predictions."""

    cer: float
    wer: float
    exact_match_rate: float
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"OCRMetrics(CER={self.cer:.4f}, WER={self.wer:.4f}, "
            f"EM={self.exact_match_rate:.4f}, n={self.n_samples})"
        )


def compute_metrics(
    predictions: list[str], references: list[str]
) -> OCRMetrics:
    """Compute aggregate OCR metrics over a list of prediction-reference pairs.

    Args:
        predictions: OCR output strings.
        references: Ground truth strings.

    Returns:
        OCRMetrics with averaged CER, WER, and exact match rate.
    """
    assert len(predictions) == len(references), (
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
    )
    n = len(predictions)
    if n == 0:
        return OCRMetrics(cer=0.0, wer=0.0, exact_match_rate=0.0, n_samples=0)

    cers = [character_error_rate(p, r) for p, r in zip(predictions, references)]
    wers = [word_error_rate(p, r) for p, r in zip(predictions, references)]
    ems = [exact_match(p, r) for p, r in zip(predictions, references)]

    return OCRMetrics(
        cer=sum(cers) / n,
        wer=sum(wers) / n,
        exact_match_rate=sum(ems) / n,
        n_samples=n,
    )


def compute_per_sample_metrics(
    predictions: list[str], references: list[str]
) -> list[dict[str, float]]:
    """Compute per-sample metrics for detailed analysis.

    Returns a list of dicts, each with keys 'cer', 'wer', 'exact_match'.
    """
    assert len(predictions) == len(references)
    results = []
    for pred, ref in zip(predictions, references):
        results.append(
            {
                "cer": character_error_rate(pred, ref),
                "wer": word_error_rate(pred, ref),
                "exact_match": exact_match(pred, ref),
            }
        )
    return results

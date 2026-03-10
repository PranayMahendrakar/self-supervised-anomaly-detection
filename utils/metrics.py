"""
Evaluation Metrics for Anomaly Detection
==========================================
Comprehensive evaluation suite for time-series anomaly detection,
including threshold-free metrics (AUROC, AUPRC) and threshold-based
metrics (F1, Precision, Recall) with point-adjust post-processing.
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnomalyDetectionResults:
    """Structured container for evaluation results."""
    auroc: float          # Area Under ROC Curve (threshold-free)
    auprc: float          # Area Under Precision-Recall Curve
    best_f1: float        # Best achievable F1 across all thresholds
    best_precision: float # Precision at best F1 threshold
    best_recall: float    # Recall at best F1 threshold
    best_threshold: float # Threshold yielding best F1
    pa_f1: float          # Point-adjusted F1 (common in literature)
    pa_precision: float
    pa_recall: float

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  Anomaly Detection Evaluation Results\n"
            f"{'='*50}\n"
            f"  AUROC:              {self.auroc:.4f}\n"
            f"  AUPRC:              {self.auprc:.4f}\n"
            f"  Best F1:            {self.best_f1:.4f}\n"
            f"  Best Precision:     {self.best_precision:.4f}\n"
            f"  Best Recall:        {self.best_recall:.4f}\n"
            f"  Best Threshold:     {self.best_threshold:.6f}\n"
            f"  PA-F1:              {self.pa_f1:.4f}\n"
            f"  PA-Precision:       {self.pa_precision:.4f}\n"
            f"  PA-Recall:          {self.pa_recall:.4f}\n"
            f"{'='*50}"
        )


def auroc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve using the trapezoidal rule.
    Measures ability to rank anomalies above normal points.

    Returns value in [0, 1]. Random detector = 0.5, perfect = 1.0.
    """
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        warnings.warn("AUROC undefined: all labels are the same class")
        return float('nan')
    return roc_auc_score(labels, scores)


def auprc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve.
    More informative than AUROC for imbalanced datasets (common in anomaly detection).
    """
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        warnings.warn("AUPRC undefined: no positive labels")
        return float('nan')
    return average_precision_score(labels, scores)


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray,
                       n_thresholds: int = 200) -> Tuple[float, float, float, float]:
    """
    Find the threshold that maximizes F1 score.

    Args:
        labels:        Ground truth binary labels (0=normal, 1=anomaly)
        scores:        Anomaly scores (higher = more anomalous)
        n_thresholds:  Number of thresholds to evaluate

    Returns:
        best_f1, best_precision, best_recall, best_threshold
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_f1, best_prec, best_rec, best_thresh = 0.0, 0.0, 0.0, 0.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1, best_prec, best_rec, best_thresh = f1, precision, recall, threshold

    return best_f1, best_prec, best_rec, best_thresh


def point_adjust(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    Point-Adjust post-processing (PA criterion).

    If ANY point in an anomaly segment is correctly detected,
    the ENTIRE segment is counted as detected.

    This is the standard evaluation protocol for time-series anomaly detection
    (used in THOC, Anomaly Transformer, etc.) as it accounts for the fact
    that models may detect anomalies slightly before/after the labeled point.

    Args:
        labels: Ground truth (0/1)
        preds:  Binary predictions (0/1)

    Returns:
        pa_preds: Point-adjusted predictions
    """
    pa_preds = preds.copy()
    in_segment = False
    seg_start = 0

    for i in range(len(labels)):
        if labels[i] == 1 and not in_segment:
            in_segment = True
            seg_start = i
        elif labels[i] == 0 and in_segment:
            # Check if any prediction in the anomaly segment was detected
            if preds[seg_start:i].max() == 1:
                pa_preds[seg_start:i] = 1
            in_segment = False

    # Handle segment at end of sequence
    if in_segment and preds[seg_start:].max() == 1:
        pa_preds[seg_start:] = 1

    return pa_preds


def evaluate(labels: np.ndarray, scores: np.ndarray,
              n_thresholds: int = 200) -> AnomalyDetectionResults:
    """
    Full evaluation pipeline for anomaly detection.

    Computes both threshold-free (AUROC, AUPRC) and threshold-based
    metrics (F1, Precision, Recall) with and without point-adjustment.

    Args:
        labels:       Ground truth binary labels, shape (N,)
        scores:       Anomaly scores from detector, shape (N,)
        n_thresholds: Number of threshold candidates

    Returns:
        AnomalyDetectionResults dataclass
    """
    labels = np.array(labels).flatten()
    scores = np.array(scores).flatten()

    assert len(labels) == len(scores), "labels and scores must have same length"

    # Threshold-free metrics
    auc_roc = auroc_score(labels, scores)
    auc_prc = auprc_score(labels, scores)

    # Best threshold-based metrics
    best_f1, best_prec, best_rec, best_thresh = best_f1_threshold(
        labels, scores, n_thresholds
    )

    # Point-adjusted metrics
    preds_at_best = (scores >= best_thresh).astype(int)
    pa_preds = point_adjust(labels, preds_at_best)
    pa_tp = ((pa_preds == 1) & (labels == 1)).sum()
    pa_fp = ((pa_preds == 1) & (labels == 0)).sum()
    pa_fn = ((pa_preds == 0) & (labels == 1)).sum()
    pa_precision = pa_tp / (pa_tp + pa_fp + 1e-8)
    pa_recall = pa_tp / (pa_tp + pa_fn + 1e-8)
    pa_f1 = 2 * pa_precision * pa_recall / (pa_precision + pa_recall + 1e-8)

    return AnomalyDetectionResults(
        auroc=float(auc_roc),
        auprc=float(auc_prc),
        best_f1=float(best_f1),
        best_precision=float(best_prec),
        best_recall=float(best_rec),
        best_threshold=float(best_thresh),
        pa_f1=float(pa_f1),
        pa_precision=float(pa_precision),
        pa_recall=float(pa_recall),
    )


def threshold_from_percentile(scores: np.ndarray, percentile: float = 95.0) -> float:
    """
    Set detection threshold as a percentile of anomaly scores on normal data.

    Args:
        scores:     Anomaly scores on validation/normal data
        percentile: Top-k% are considered anomalous (e.g., 95 = top 5%)
    """
    return float(np.percentile(scores, percentile))


def compute_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> Dict[str, int]:
    """Compute TP, FP, TN, FN for binary classification."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def print_evaluation_report(labels: np.ndarray, scores: np.ndarray,
                              model_name: str = "Model") -> AnomalyDetectionResults:
    """
    Evaluate and print a formatted report.

    Args:
        labels:     Ground truth labels
        scores:     Anomaly scores
        model_name: Name to display in report

    Returns:
        AnomalyDetectionResults
    """
    print(f"\nEvaluating {model_name}...")
    results = evaluate(labels, scores)
    print(f"[{model_name}]{results}")
    return results

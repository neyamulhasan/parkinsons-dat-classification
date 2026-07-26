"""Binary classification metrics for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    """Standard binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def compute_binary_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float],
) -> BinaryClassificationMetrics:
    """Compute the standard metrics used by the baseline training loop."""

    true_labels = np.asarray(list(y_true), dtype=np.int32)
    predicted_labels = np.asarray(list(y_pred), dtype=np.int32)
    scores = np.asarray(list(y_score), dtype=np.float32)

    if true_labels.size == 0:
        raise ValueError("Metrics require at least one sample")

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
    accuracy = float(np.mean(true_labels == predicted_labels))
    precision = float(precision_score(true_labels, predicted_labels, zero_division=0))
    recall = float(recall_score(true_labels, predicted_labels, zero_division=0))
    f1 = float(f1_score(true_labels, predicted_labels, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(true_labels, scores))
    except ValueError:
        roc_auc = 0.0

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )

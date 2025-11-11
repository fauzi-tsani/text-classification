import numpy as np
from typing import Dict, List, Tuple


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report_np(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]
) -> Dict:
    num_classes = len(label_names)
    report = {}
    total = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    report["accuracy"] = correct / total if total > 0 else 0.0

    weighted_f1_sum = 0.0
    weighted_support_sum = 0

    for idx, name in enumerate(label_names):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        support = int(np.sum(y_true == idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

        weighted_f1_sum += f1 * support
        weighted_support_sum += support

    report["weighted avg"] = {
        "precision": np.average([report[name]["precision"] for name in label_names], weights=[report[name]["support"] for name in label_names]) if weighted_support_sum > 0 else 0.0,
        "recall": np.average([report[name]["recall"] for name in label_names], weights=[report[name]["support"] for name in label_names]) if weighted_support_sum > 0 else 0.0,
        "f1-score": weighted_f1_sum / weighted_support_sum if weighted_support_sum > 0 else 0.0,
        "support": weighted_support_sum,
    }

    return report


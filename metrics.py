"""Single-label multiclass metrics with one-vs-rest AUC reporting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def compute_metrics(labels, probs, class_names):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    n_classes = len(class_names)
    if labels.ndim != 1 or probs.ndim != 2 or probs.shape != (labels.size, n_classes):
        raise ValueError(f"Expected labels [N] and probabilities [N,{n_classes}], got {labels.shape} and {probs.shape}")
    if labels.size == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("Evaluation labels contain an index outside class_names")
    if not np.isfinite(probs).all() or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Class probabilities must be finite values in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Each class-probability row must sum to 1")
    preds = probs.argmax(axis=1)
    class_ids = np.arange(n_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=class_ids, zero_division=0)
    y_bin = np.eye(n_classes, dtype=np.int64)[labels]
    per_auc = {}
    valid_auc = []
    for idx, name in enumerate(class_names):
        try:
            auc = float(roc_auc_score(y_bin[:, idx], probs[:, idx]))
        except ValueError:
            auc = None
        per_auc[name] = auc
        if auc is not None:
            valid_auc.append(auc)
    try:
        auc_macro = float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
        auc_micro = float(roc_auc_score(y_bin, probs, average="micro", multi_class="ovr"))
    except ValueError:
        auc_macro = float(np.mean(valid_auc)) if valid_auc else None
        try:
            auc_micro = float(roc_auc_score(y_bin.ravel(), probs.ravel()))
        except ValueError:
            auc_micro = None
    brier_score = float(np.mean(np.sum((probs - y_bin) ** 2, axis=1)))
    confidence = probs.max(axis=1)
    correct = preds == labels
    bin_edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for bin_index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if bin_index == len(bin_edges) - 2:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        if np.any(in_bin):
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidence[in_bin].mean()))
    return {
        "sample_count": int(labels.size),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_recall_fscore_support(labels, preds, labels=class_ids, average="macro", zero_division=0)[0]),
        "precision_micro": float(precision_recall_fscore_support(labels, preds, labels=class_ids, average="micro", zero_division=0)[0]),
        "recall_macro": float(precision_recall_fscore_support(labels, preds, labels=class_ids, average="macro", zero_division=0)[1]),
        "recall_micro": float(precision_recall_fscore_support(labels, preds, labels=class_ids, average="micro", zero_division=0)[1]),
        "f1_macro": float(f1_score(labels, preds, labels=class_ids, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, preds, labels=class_ids, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, labels=class_ids, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": auc_macro,
        "roc_auc_ovr_micro": auc_micro,
        "brier_score_multiclass": brier_score,
        "ece_top_label": ece,
        "ece_bin_count": 15,
        "ece_bin_strategy": "15 equal-width confidence bins over [0,1]; top-label correctness",
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i]),
                              "roc_auc_ovr": per_auc[name]} for i, name in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(labels, preds, labels=np.arange(n_classes)).tolist(),
    }

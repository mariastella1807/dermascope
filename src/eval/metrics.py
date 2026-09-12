"""Metricas exigidas por el enunciado.

§4.1 clasificacion: accuracy y F1. Se reporta macro-F1 como metrica principal porque el
dataset es extremadamente desbalanceado (nv ~6.705 vs df ~115): un modelo que prediga
siempre `nv` alcanza ~67% de accuracy sin haber aprendido nada. El macro-F1 promedia por
clase y castiga ese comportamiento.

§4.2 segmentacion: Dice e IoU, calculados por imagen y promediados. El promedio por
imagen (macro) y no sobre el total de pixeles (micro) evita que las lesiones grandes
dominen el resultado.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class_f1": {
            name: float(score)
            for name, score in zip(
                class_names,
                f1_score(y_true, y_pred, average=None, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        ),
    }


def dice_coefficient(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7
) -> torch.Tensor:
    """Dice por imagen. Entrada: logits (B,1,H,W) y target binario (B,1,H,W)."""
    pred = (torch.sigmoid(logits) > threshold).float()
    target = (target > 0.5).float()
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    # Si ambas son vacias el acuerdo es perfecto: Dice = 1.
    return (2 * intersection + eps) / (denominator + eps)


def iou_score(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7
) -> torch.Tensor:
    """IoU (Jaccard) por imagen."""
    pred = (torch.sigmoid(logits) > threshold).float()
    target = (target > 0.5).float()
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    return (intersection + eps) / (union + eps)


def bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.5
) -> torch.Tensor:
    """BCE + Dice soft.

    La BCE sola sesga hacia el fondo, que ocupa la mayor parte del area en muchas
    imagenes; el termino Dice optimiza directamente el solapamiento y estabiliza el
    entrenamiento en las lesiones pequenas.
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    soft_dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return (1 - dice_weight) * bce + dice_weight * (1 - soft_dice.mean())


def class_weights_balanced(counts: np.ndarray) -> torch.Tensor:
    """Pesos inversos a la frecuencia, normalizados a media 1.

    La normalizacion mantiene la magnitud de la perdida comparable a la version sin
    pesos, de modo que el learning rate del YAML sigue siendo valido.
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

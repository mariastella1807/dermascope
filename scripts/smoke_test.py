"""Verificacion del entorno sin necesidad del dataset.

Comprueba que todo lo que el proyecto importa existe y que los dos modelos se
construyen y producen tensores de la forma esperada. Correr esto despues de crear el
entorno y cada vez que se actualicen dependencias: detecta cambios de API en
transformers, albumentations o torch antes de que aparezcan a mitad de un
entrenamiento en Colab.

Uso:
    .venv\\Scripts\\python.exe scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PASSED, FAILED = [], []


def check(name: str):
    """Decorador que ejecuta una comprobacion y registra el resultado."""

    def wrapper(fn):
        try:
            detail = fn()
            PASSED.append((name, detail or "ok"))
        except Exception as exc:  # noqa: BLE001 - el objetivo es reportar, no propagar
            FAILED.append((name, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        return fn

    return wrapper


@check("versiones")
def _versions() -> str:
    import albumentations
    import numpy
    import pandas
    import torch
    import transformers

    return (
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"albumentations {albumentations.__version__} | pandas {pandas.__version__} | "
        f"numpy {numpy.__version__} | cuda {torch.cuda.is_available()}"
    )


@check("config")
def _config() -> str:
    from src.config import load_config

    cfg = load_config("configs/classification.yaml")
    assert cfg.model.num_classes == len(cfg.classes), "num_classes no coincide con classes"
    assert cfg.seed == 42
    return f"{len(cfg.classes)} clases, semilla {cfg.seed}, herencia de paths.yaml ok"


@check("transforms")
def _transforms() -> str:
    import numpy as np

    from src.config import load_config
    from src.data import transforms as T

    cls_cfg = load_config("configs/classification.yaml")
    seg_cfg = load_config("configs/segmentation.yaml")
    image = np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
    mask = (np.random.rand(450, 600) > 0.5).astype(np.float32)

    out = T.classification_train_transform(cls_cfg)(image=image)["image"]
    size = cls_cfg.data.image_size
    assert tuple(out.shape) == (3, size, size), f"forma inesperada: {out.shape}"

    T.classification_eval_transform(cls_cfg)(image=image)
    seg_out = T.segmentation_train_transform(seg_cfg)(image=image, mask=mask)
    seg_size = seg_cfg.data.image_size
    assert tuple(seg_out["image"].shape) == (3, seg_size, seg_size)
    assert seg_out["mask"].shape[-2:] == (seg_size, seg_size)
    return f"clasificacion {size}px, segmentacion {seg_size}px, mascara alineada"


@check("clasificador + CBAM")
def _classifier() -> str:
    import torch

    from src.config import load_config
    from src.models.classifier import build_classifier

    cfg = load_config("configs/classification.yaml")
    with_cbam = build_classifier(cfg, override_cbam=True).eval()
    without = build_classifier(cfg, override_cbam=False).eval()

    x = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size)
    with torch.no_grad():
        y = with_cbam(x)
    assert tuple(y.shape) == (2, cfg.model.num_classes), f"logits: {y.shape}"

    n_cbam = sum(p.numel() for p in with_cbam.parameters())
    n_plain = sum(p.numel() for p in without.parameters())
    assert n_cbam > n_plain, "CBAM no anadio parametros: no se inyecto"

    groups = with_cbam.param_groups(3e-4, 10.0)
    assert len(groups) == 2 and all(g["params"] for g in groups), "param_groups vacio"
    return (
        f"sin CBAM {n_plain / 1e6:.2f}M -> con CBAM {n_cbam / 1e6:.2f}M "
        f"(+{(n_cbam - n_plain) / 1e3:.0f}K params)"
    )


@check("Grad-CAM")
def _gradcam() -> str:
    import torch

    from src.config import load_config
    from src.explain.gradcam import GradCAM, SmoothGradCAM
    from src.models.classifier import build_classifier

    cfg = load_config("configs/classification.yaml")
    model = build_classifier(cfg).eval()
    x = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)

    with GradCAM(model, model.gradcam_target_layer) as cam_fn:
        cam, idx = cam_fn(x)
    assert cam.shape == (cfg.data.image_size, cfg.data.image_size), f"cam: {cam.shape}"
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-5, "cam fuera de [0,1]"

    with SmoothGradCAM(model, model.gradcam_target_layer) as smooth_fn:
        smooth, _ = smooth_fn(x, idx, n_samples=2)
    assert smooth.shape == cam.shape
    return f"mapa {cam.shape}, clase {idx}, rango [{cam.min():.3f}, {cam.max():.3f}]"


@check("aislamiento de la lesion")
def _isolate() -> str:
    import numpy as np

    from src.explain.isolate import isolate_lesion

    image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    prob = np.zeros((200, 300), dtype=np.float32)
    prob[60:140, 100:200] = 0.9          # lesion principal
    prob[10:16, 10:16] = 0.9             # componente satelite que debe descartarse

    result = isolate_lesion(image, prob, keep_largest_component=True)
    assert result.detected, "no detecto la lesion principal"
    assert result.rgba.shape == (200, 300, 4), f"rgba: {result.rgba.shape}"
    assert result.mask[12, 12] == 0, "no descarto la componente satelite"
    assert result.mask[100, 150] == 1, "descarto la lesion principal"
    x, y, w, h = result.bbox
    return f"bbox=({x},{y},{w},{h}), area={result.area_ratio:.1%}, satelite descartado"


@check("metricas")
def _metrics() -> str:
    import numpy as np
    import torch

    from src.eval.metrics import (
        bce_dice_loss,
        class_weights_balanced,
        classification_metrics,
        dice_coefficient,
        iou_score,
    )

    names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 5, 5, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 5, 5, 4])
    m = classification_metrics(y_true, y_pred, names)
    assert 0 <= m["macro_f1"] <= 1

    # Prediccion perfecta: Dice e IoU deben valer 1.
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:24, 8:24] = 1.0
    logits = torch.where(target > 0.5, 10.0, -10.0)
    dice = dice_coefficient(logits, target).mean().item()
    iou = iou_score(logits, target).mean().item()
    assert dice > 0.99 and iou > 0.99, f"dice={dice} iou={iou}"
    assert bce_dice_loss(logits, target).item() < 0.01

    weights = class_weights_balanced(np.array([6705, 1113, 1099, 514, 327, 115, 142]))
    assert weights[5] > weights[0], "el peso de la clase rara deberia ser mayor"
    return (
        f"macro_f1={m['macro_f1']:.3f}, dice={dice:.4f}, iou={iou:.4f}, "
        f"pesos ok (rango {weights.min():.2f}-{weights.max():.2f})"
    )


@check("segmentador SegFormer")
def _segformer() -> str:
    import torch

    from src.config import load_config
    from src.models.segmenter import build_segmenter

    cfg = load_config("configs/segmentation.yaml")
    model = build_segmenter(cfg).eval()
    size = cfg.data.image_size
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1, size, size), f"logits: {y.shape}"
    n = sum(p.numel() for p in model.parameters())
    return f"salida {tuple(y.shape)}, {n / 1e6:.2f}M params (descarga pesos de HuggingFace)"


@check("baseline U-Net")
def _unet() -> str:
    import torch

    from src.config import load_config
    from src.models.segmenter import build_segmenter

    cfg = load_config("configs/segmentation.yaml")
    cfg["model"]["arch"] = "unet"
    model = build_segmenter(cfg).eval()
    size = cfg.data.image_size
    with torch.no_grad():
        y = model(torch.randn(1, 3, size, size))
    assert tuple(y.shape) == (1, 1, size, size), f"logits: {y.shape}"
    n = sum(p.numel() for p in model.parameters())
    return f"salida {tuple(y.shape)}, {n / 1e6:.2f}M params"


def main() -> int:
    print("=" * 72)
    for name, detail in PASSED:
        print(f"  OK    {name:26s} {detail}")
    for name, message, _ in FAILED:
        print(f"  FALLA {name:26s} {message}")
    print("=" * 72)
    print(f"{len(PASSED)} pasaron, {len(FAILED)} fallaron")

    if FAILED:
        print("\n--- Detalle de los fallos ---")
        for name, _, tb in FAILED:
            print(f"\n### {name}\n{tb}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

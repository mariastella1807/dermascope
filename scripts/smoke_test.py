"""Verificacion del entorno sin necesidad del dataset.

Comprueba que todo lo que el proyecto importa existe y que los dos modelos se
construyen y producen tensores de la forma esperada. Correr esto despues de crear el
entorno y cada vez que se actualicen dependencias: detecta cambios de API en torch o
torchvision antes de que aparezcan a mitad de un entrenamiento en Colab.

La ultima comprobacion verifica que el proyecto no importe librerias ajenas a su conjunto
de dependencias (albumentations, opencv, transformers, segmentation-models-pytorch).

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
    import numpy
    import pandas
    import scipy
    import torch
    import torchvision

    return (
        f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
        f"scipy {scipy.__version__} | pandas {pandas.__version__} | "
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
    import torch
    from PIL import Image
    from torchvision import tv_tensors

    from src.config import load_config
    from src.data import transforms as T

    cls_cfg = load_config("configs/classification.yaml")
    seg_cfg = load_config("configs/segmentation.yaml")
    array = np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)

    size = cls_cfg.data.image_size
    out = T.classification_train_transform(cls_cfg)(Image.fromarray(array))
    assert tuple(out.shape) == (3, size, size), f"forma inesperada: {out.shape}"
    assert out.dtype == torch.float32
    # Streamlit entrega arreglos NumPy: la transformacion de evaluacion debe aceptarlos.
    assert tuple(T.classification_eval_transform(cls_cfg)(array).shape) == (3, size, size)

    # Imagen con la mitad izquierda blanca y mascara que marca esa misma mitad. Despues de
    # rotar, escalar y voltear al azar, los pixeles blancos deben seguir coincidiendo con
    # la mascara: si no, la geometria se aplico distinto a cada una.
    seg_size = seg_cfg.data.image_size
    half = np.zeros((450, 600, 3), dtype=np.uint8)
    half[:, :300] = 255
    mask = torch.zeros(450, 600, dtype=torch.uint8)
    mask[:, :300] = 1
    mask = tv_tensors.Mask(mask)
    seg_tf = T.segmentation_train_transform(seg_cfg)
    worst = 1.0
    for _ in range(10):
        img_t, mask_t = seg_tf(Image.fromarray(half), mask)
        assert tuple(img_t.shape) == (3, seg_size, seg_size)
        assert tuple(mask_t.shape) == (seg_size, seg_size)
        assert set(mask_t.unique().tolist()) <= {0, 1}, "la mascara dejo de ser binaria"
        bright = img_t.mean(dim=0) > 0.5
        agreement = (bright == mask_t.bool()).float().mean().item()
        worst = min(worst, agreement)
    assert worst > 0.95, f"imagen y mascara desalineadas ({worst:.1%})"
    return f"clasificacion {size}px, segmentacion {seg_size}px, alineacion minima {worst:.1%}"


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


@check("self-attention")
def _self_attention() -> str:
    import torch

    from src.models.attention import SelfAttention2d

    torch.manual_seed(0)
    block = SelfAttention2d(channels=32, n_tokens=16)
    x = torch.randn(2, 32, 4, 4)
    y = block(x)
    assert y.shape == x.shape, f"forma: {y.shape}"
    # La proyeccion de salida arranca en ceros: al inicio el bloque es la identidad.
    assert torch.allclose(y, x), "el bloque sin entrenar deberia devolver la entrada"
    weights = block.last_attention
    assert tuple(weights.shape) == (2, 16, 16), f"atencion: {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 16)), "las filas no suman 1"

    # Estabilidad en float16: con Q y K grandes, Q K^T supera 65.504 (maximo de float16).
    # El bloque calcula los puntajes en float32, asi que la salida debe seguir siendo finita.
    grande = SelfAttention2d(channels=32, n_tokens=16).half()
    with torch.no_grad():
        grande.query.weight.fill_(2.0)
        grande.key.weight.fill_(2.0)
        grande.proj.weight.normal_(std=0.02)
    salida = grande(torch.randn(2, 32, 4, 4).half())
    producto_fp16 = (torch.full((1, 32), 300.0).half() @ torch.full((32, 1), 300.0).half()).item()
    assert producto_fp16 == float("inf"), "el ejemplo deberia desbordar en float16"
    assert torch.isfinite(salida).all(), "la atencion produjo NaN o infinitos en float16"
    return "softmax(QK^T/sqrt(d))V: filas suman 1, identidad al inicio, estable en float16"


@check("U-Net con y sin self-attention")
def _unet() -> str:
    import torch

    from src.config import load_config
    from src.models.segmenter import build_segmenter

    cfg = load_config("configs/segmentation.yaml")
    size = cfg.data.image_size
    x = torch.randn(1, 3, size, size)
    counts = {}
    for use_attention in (True, False):
        model = build_segmenter(cfg, override_attention=use_attention).eval()
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (1, 1, size, size), f"logits: {y.shape}"
        counts[use_attention] = sum(p.numel() for p in model.parameters())
        groups = model.param_groups(1e-4, 10.0)
        assert len(groups) == 2 and all(g["params"] for g in groups), "param_groups vacio"
    assert counts[True] > counts[False], "el bloque de atencion no anadio parametros"
    return (
        f"salida {tuple(y.shape)}, sin atencion {counts[False] / 1e6:.2f}M -> "
        f"con atencion {counts[True] / 1e6:.2f}M"
    )


@check("ablaciones controladas")
def _controlled_ablation() -> str:
    """Con la misma semilla, lo comun a las dos corridas debe arrancar identico.

    Si el bloque de atencion se inicializara antes que las capas nuevas comunes, estas
    arrancarian con pesos distintos y la ablacion no aislaria el efecto del bloque.
    """
    import torch

    from src.config import load_config, set_seed
    from src.models.classifier import build_classifier
    from src.models.segmenter import build_segmenter

    cls_cfg = load_config("configs/classification.yaml")
    seg_cfg = load_config("configs/segmentation.yaml")

    def construir(fabrica):
        set_seed(42)
        return fabrica()

    con = construir(lambda: build_classifier(cls_cfg, override_cbam=True))
    sin = construir(lambda: build_classifier(cls_cfg, override_cbam=False))
    assert torch.equal(con.net.fc[1].weight, sin.net.fc[1].weight), "la capa final arranca distinta con y sin CBAM"

    con = construir(lambda: build_segmenter(seg_cfg, override_attention=True))
    sin = construir(lambda: build_segmenter(seg_cfg, override_attention=False))
    comunes = [n for n, _ in sin.named_parameters()]
    params_con = dict(con.named_parameters())
    distintos = [n for n, p in sin.named_parameters() if not torch.equal(p, params_con[n])]
    assert not distintos, f"la U-Net arranca distinta con y sin atencion en: {distintos[:3]}"
    return f"capa final del clasificador y {len(comunes)} tensores comunes de la U-Net identicos"


@check("dependencias permitidas")
def _no_external_libraries() -> str:
    import importlib
    import pkgutil

    import src

    # Importa todos los modulos del proyecto y del app, y revisa que ninguno haya
    # arrastrado una libreria ajena a las dependencias declaradas.
    for module in pkgutil.walk_packages(src.__path__, prefix="src."):
        importlib.import_module(module.name)
    importlib.import_module("app.inference")

    forbidden = ["albumentations", "cv2", "transformers", "segmentation_models_pytorch"]
    loaded = [name for name in forbidden if name in sys.modules]
    assert not loaded, f"se importaron: {loaded}"
    return "ninguna de: " + ", ".join(forbidden)


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

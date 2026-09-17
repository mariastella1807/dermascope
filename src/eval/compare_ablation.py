"""Consolida las ablaciones de atencion en las tablas del informe (§4.3).

§4.3 no se satisface mostrando que los modelos con atencion existen: exige explicar que
aporta cada mecanismo frente a una version sin el. Este script produce las dos tablas:

1. CBAM: clasificador con y sin el bloque -> accuracy, macro-F1, F1 por clase, parametros.
2. Self-attention: U-Net con y sin el bloque -> Dice, IoU, parametros.

Ademas calcula la metrica de localizacion de la atencion: que fraccion de la masa de
Grad-CAM cae dentro de la mascara real de la lesion. Ese numero convierte el analisis
cualitativo de los mapas en evidencia cuantitativa, que es mas defendible en la
sustentacion que "se ve mejor".

Uso:
    python -m src.eval.compare_ablation
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from src.config import load_config, resolve


def load_results(results_dir, filenames: list[str]) -> dict[str, dict]:
    out = {}
    for name in filenames:
        path = results_dir / name
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"Falta {name}: corre la corrida correspondiente antes de comparar.")
    return out


def cbam_table(results: dict[str, dict], class_names: list[str]) -> pd.DataFrame:
    rows = []
    for filename, label in [
        ("classification_nocbam.json", "ResNet-34 (sin CBAM)"),
        ("classification_cbam.json", "ResNet-34 + CBAM"),
    ]:
        data = results.get(filename)
        if not data:
            continue
        test = data["test"]
        row = {
            "modelo": label,
            "params_M": round(data["n_params"] / 1e6, 2),
            "tamano_MB": round(data["checkpoint_mb"], 2),
            "accuracy": round(test["accuracy"], 4),
            "balanced_acc": round(test["balanced_accuracy"], 4),
            "macro_F1": round(test["macro_f1"], 4),
        }
        for cls in class_names:
            row[f"F1_{cls}"] = round(test["per_class_f1"].get(cls, float("nan")), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def segmentation_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for filename, label in [
        ("segmentation_noattn.json", "U-Net ResNet-34 (sin self-attention)"),
        ("segmentation_attn.json", "U-Net ResNet-34 + self-attention"),
    ]:
        data = results.get(filename)
        if not data:
            continue
        rows.append(
            {
                "modelo": label,
                "params_M": round(data["n_params"] / 1e6, 2),
                "Dice": round(data["test_dice"], 4),
                "IoU": round(data["test_iou"], 4),
                "min_entrenamiento": round(data["train_minutes"], 1),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def _predict_probs(model, tensor):
    return torch.sigmoid(model(tensor))[0, 0].cpu().numpy()


def attention_localization(cfg_cls, n_images: int = 100) -> pd.DataFrame:
    """Fraccion de la masa de Grad-CAM dentro de la mascara real, con y sin CBAM.

    Se evalua solo sobre imagenes de test que tienen mascara anotada, porque la mascara
    de referencia es lo que define "dentro de la lesion".
    """
    from src.data import transforms as T
    from src.data.datasets import load_splits, read_mask, read_rgb
    from src.explain.gradcam import GradCAM, attention_mass_in_mask
    from src.models.classifier import build_classifier

    splits = load_splits(cfg_cls)
    subset = splits[(splits["split"] == "test") & (splits["has_mask"])].head(n_images)
    if subset.empty:
        print("No hay imagenes de test con mascara: se omite la metrica de localizacion.")
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_tf = T.classification_eval_transform(cfg_cls)
    models_dir = resolve(cfg_cls.paths.models)

    rows = []
    for tag, use_cbam in [("nocbam", False), ("cbam", True)]:
        ckpt = models_dir / f"classifier_{tag}_best.pt"
        if not ckpt.exists():
            print(f"Falta {ckpt.name}: se omite {tag} en la metrica de localizacion.")
            continue

        model = build_classifier(cfg_cls, override_cbam=use_cbam).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])

        fractions = []
        with GradCAM(model, model.gradcam_target_layer) as cam_fn:
            for _, row in subset.iterrows():
                tensor = eval_tf(read_rgb(row["image_path"])).unsqueeze(0).to(device)
                cam, _ = cam_fn(tensor)

                # La mascara original (600x450) se lleva al tamano del mapa (224x224) con
                # vecino mas cercano, para que siga siendo binaria.
                mask = torch.from_numpy(read_mask(row["mask_path"])).float()[None, None]
                mask = torch.nn.functional.interpolate(mask, size=cam.shape, mode="nearest")
                fractions.append(attention_mass_in_mask(cam, mask[0, 0].numpy()))

        rows.append(
            {
                "modelo": "ResNet-34 + CBAM" if use_cbam else "ResNet-34 (sin CBAM)",
                "n_imagenes": len(fractions),
                "masa_atencion_en_lesion_media": round(float(np.mean(fractions)), 4),
                "masa_atencion_en_lesion_p50": round(float(np.median(fractions)), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cls-config", default="configs/classification.yaml")
    parser.add_argument(
        "--skip-localization",
        action="store_true",
        help="Omite la metrica de Grad-CAM, que requiere ambos checkpoints",
    )
    args = parser.parse_args()

    cfg = load_config(args.cls_config)
    results_dir = resolve(cfg.paths.reports) / "results"
    class_names = list(cfg.classes)

    results = load_results(
        results_dir,
        [
            "classification_cbam.json",
            "classification_nocbam.json",
            "segmentation_attn.json",
            "segmentation_noattn.json",
        ],
    )

    tables = {
        "ablacion_cbam": cbam_table(results, class_names),
        "comparacion_segmentacion": segmentation_table(results),
    }
    if not args.skip_localization:
        tables["localizacion_atencion"] = attention_localization(cfg)

    for name, table in tables.items():
        if table.empty:
            continue
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))
        table.to_csv(results_dir / f"{name}.csv", index=False)

    print(f"\nTablas escritas en {results_dir}")


if __name__ == "__main__":
    main()

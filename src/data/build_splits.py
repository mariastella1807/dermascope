"""Construccion de los splits train/val/test.

Dos decisiones que definen la validez de todas las metricas del proyecto:

1. El agrupamiento se hace por `lesion_id`, no por imagen. HAM10000 contiene varias
   fotografias de la misma lesion fisica (~7.470 lesiones para 10.015 imagenes). Si
   dos fotos de la misma lesion caen en train y en test, el modelo la reconoce en vez
   de generalizar, y el accuracy reportado deja de significar nada.
2. La estratificacion se hace por diagnostico a nivel de lesion, para que las clases
   raras (`df`, `vasc`) aparezcan en los tres splits.

Salida: `data/processed/splits.csv` con columnas
`image_id, lesion_id, dx, dx_idx, split, has_mask, image_path, mask_path`.

Uso:
    python -m src.data.build_splits --config configs/paths.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from src.config import load_config, resolve, set_seed


def assign_groups(
    lesions: pd.DataFrame, ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Reparte `lesion_id` en splits respetando las proporciones dentro de cada clase.

    Se asigna por clase y de forma deterministica (barajado con semilla fija) en vez de
    usar train_test_split dos veces: con 7 clases muy desbalanceadas, el reparto
    secuencial deja las clases raras sin representacion en val o test.
    """
    assignment: dict[str, str] = {}

    for dx, group in lesions.groupby("dx", sort=True):
        ids = group["lesion_id"].sample(frac=1.0, random_state=seed).tolist()
        n = len(ids)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        # El resto va a test, para que los tres subconjuntos sumen exactamente n.
        n_val = min(n_val, n - n_train)

        for i, lesion_id in enumerate(ids):
            if i < n_train:
                assignment[lesion_id] = "train"
            elif i < n_train + n_val:
                assignment[lesion_id] = "val"
            else:
                assignment[lesion_id] = "test"

    return assignment


def build(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    metadata_path = resolve(cfg.paths.ham_metadata)
    images_dir = resolve(cfg.paths.ham_images)
    masks_dir = resolve(cfg.paths.seg_masks)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encontro {metadata_path}. Ver scripts/download_data.md."
        )

    meta = pd.read_csv(metadata_path)
    classes = list(cfg.classes)
    dx_to_idx = {dx: i for i, dx in enumerate(classes)}

    unknown = set(meta["dx"].unique()) - set(classes)
    if unknown:
        raise ValueError(f"Diagnosticos no declarados en configs/paths.yaml: {unknown}")

    # Una fila por lesion para el agrupamiento. Todas las imagenes de una lesion
    # comparten diagnostico, asi que `first` es seguro.
    lesions = meta.groupby("lesion_id", as_index=False)["dx"].first()
    assignment = assign_groups(lesions, dict(cfg.split), cfg.seed)

    meta["split"] = meta["lesion_id"].map(assignment)
    meta["dx_idx"] = meta["dx"].map(dx_to_idx)

    # Cruce con las mascaras de ISIC 2018 Task 1 por el identificador ISIC_xxxxxxx.
    # El nombre de archivo de la mascara es ISIC_xxxxxxx_segmentation.png.
    mask_paths = {}
    if masks_dir.exists():
        for mask in masks_dir.rglob("*.png"):
            image_id = mask.stem.replace("_segmentation", "")
            mask_paths[image_id] = mask

    meta["image_path"] = meta["image_id"].map(
        lambda i: str(_find_image(images_dir, i))
    )
    meta["mask_path"] = meta["image_id"].map(
        lambda i: str(mask_paths[i]) if i in mask_paths else ""
    )
    meta["has_mask"] = meta["mask_path"] != ""

    out_dir = resolve(cfg.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_id",
        "lesion_id",
        "dx",
        "dx_idx",
        "split",
        "has_mask",
        "image_path",
        "mask_path",
    ]
    splits = meta[columns].sort_values("image_id").reset_index(drop=True)
    splits.to_csv(out_dir / "splits.csv", index=False)

    _report(splits)
    return splits


def _find_image(images_dir, image_id: str):
    """HAM10000 se distribuye en dos carpetas (part_1 y part_2) segun la fuente."""
    for candidate in (
        images_dir / f"{image_id}.jpg",
        images_dir / "HAM10000_images_part_1" / f"{image_id}.jpg",
        images_dir / "HAM10000_images_part_2" / f"{image_id}.jpg",
    ):
        if candidate.exists():
            return candidate
    # Ultimo recurso: busqueda recursiva.
    matches = list(images_dir.rglob(f"{image_id}.jpg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Imagen ausente para image_id={image_id} en {images_dir}")


def _report(splits: pd.DataFrame) -> None:
    print("\n=== Imagenes por split y clase ===")
    print(pd.crosstab(splits["dx"], splits["split"], margins=True))

    print("\n=== Lesiones unicas por split ===")
    print(splits.groupby("split")["lesion_id"].nunique())

    print("\n=== Cobertura de mascaras (subconjunto de segmentacion) ===")
    with_mask = splits[splits["has_mask"]]
    print(f"Imagenes con mascara: {len(with_mask)} de {len(splits)}")
    if len(with_mask):
        print(pd.crosstab(with_mask["dx"], with_mask["split"], margins=True))

    # Verificacion de fuga: ninguna lesion debe aparecer en dos splits.
    leaked = (
        splits.groupby("lesion_id")["split"].nunique().loc[lambda s: s > 1].index.tolist()
    )
    if leaked:
        raise AssertionError(f"Fuga de lesiones entre splits: {leaked[:10]}")
    print("\nOK: ninguna lesion aparece en mas de un split.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()

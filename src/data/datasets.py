"""Datasets de PyTorch para las dos tareas, ambos leyendo de `data/processed/splits.csv`.

Que las dos tareas partan del mismo CSV es lo que garantiza que comparten el split por
`lesion_id`: una lesion que esta en test para clasificacion tambien esta en test para
segmentacion, asi que el pipeline completo del app nunca ve en inferencia una lesion que
alguno de los dos modelos uso para entrenar.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import resolve


def load_splits(cfg) -> pd.DataFrame:
    path = resolve(cfg.paths.processed) / "splits.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre primero: python -m src.data.build_splits"
        )
    return pd.read_csv(path)


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class DermaClassificationDataset(Dataset):
    """Imagen dermatoscopica -> indice de clase (0..6)."""

    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = _read_rgb(row["image_path"])
        image = self.transform(image=image)["image"]
        return image, int(row["dx_idx"])

    def class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for idx, n in self.frame["dx_idx"].value_counts().items():
            counts[int(idx)] = n
        return counts


class DermaSegmentationDataset(Dataset):
    """Imagen -> mascara binaria de la lesion. Solo filas con `has_mask`."""

    def __init__(self, frame: pd.DataFrame, transform) -> None:
        frame = frame[frame["has_mask"]].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                "Ninguna fila tiene mascara. Verifica data/raw/ISIC2018_Task1_masks."
            )
        self.frame = frame
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        image = _read_rgb(row["image_path"])
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"No se pudo leer la mascara: {row['mask_path']}")
        # Las mascaras de ISIC vienen en 0/255; se normalizan a 0/1 float.
        mask = (mask > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        mask_tensor = out["mask"]
        if not torch.is_tensor(mask_tensor):
            mask_tensor = torch.from_numpy(mask_tensor)
        return out["image"], mask_tensor.unsqueeze(0).float()


def make_dataloaders(cfg, task: str):
    """Construye los tres DataLoader para `task` in {"classification", "segmentation"}."""
    from torch.utils.data import DataLoader

    from src.data import transforms as T

    splits = load_splits(cfg)

    if task == "classification":
        ds_cls = DermaClassificationDataset
        train_tf = T.classification_train_transform(cfg)
        eval_tf = T.classification_eval_transform(cfg)
    elif task == "segmentation":
        ds_cls = DermaSegmentationDataset
        train_tf = T.segmentation_train_transform(cfg)
        eval_tf = T.segmentation_eval_transform(cfg)
    else:
        raise ValueError(f"Tarea desconocida: {task}")

    loaders = {}
    datasets = {}
    for split in ("train", "val", "test"):
        frame = splits[splits["split"] == split]
        dataset = ds_cls(frame, train_tf if split == "train" else eval_tf)
        datasets[split] = dataset
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders, datasets

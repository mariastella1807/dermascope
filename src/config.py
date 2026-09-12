"""Carga de configuracion YAML con resolucion de rutas relativas a la raiz del repo."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """Diccionario con acceso por punto, para leer `cfg.model.backbone`."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path) -> Config:
    """Lee un YAML y, si declara `paths_config`, fusiona ese archivo como base.

    Asi `configs/classification.yaml` hereda rutas, semilla y clases de
    `configs/paths.yaml` sin duplicarlas.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    base_ref = data.pop("paths_config", None)
    if base_ref is not None:
        base = dict(load_config(base_ref))
        base.update(data)
        data = base

    return Config(data)


def resolve(path_value: str | Path) -> Path:
    """Convierte una ruta del YAML en ruta absoluta anclada a la raiz del repo."""
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    """Fija las semillas de random, numpy y torch.

    Necesario para que la ablacion con/sin CBAM (§4.3) sea comparable: si las dos
    corridas no comparten inicializacion y orden de batches, la diferencia medida
    mezcla el efecto del bloque con el ruido de entrenamiento.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

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


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusiona `override` sobre `base` recursivamente.

    La fusion profunda importa para que un archivo local pueda cambiar una sola ruta
    sin tener que repetir el bloque `paths` completo.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> Config:
    """Lee un YAML, resuelve la herencia y aplica el override local si existe.

    Tres capas, de menor a mayor prioridad:

    1. El archivo referenciado en `paths_config`, si lo hay. Asi
       `configs/classification.yaml` hereda rutas, semilla y clases de
       `configs/paths.yaml` sin duplicarlas.
    2. El archivo pedido.
    3. `<nombre>.local.yaml` junto al archivo pedido, si existe. Estos archivos estan
       en .gitignore: sirven para que cada integrante del equipo apunte a sus propias
       rutas de datos sin ensuciar el repositorio ni generar conflictos.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    base_ref = data.pop("paths_config", None)
    if base_ref is not None:
        data = _deep_merge(dict(load_config(base_ref)), data)

    # El guardia evita buscar paths.local.local.yaml al cargar paths.local.yaml.
    if not path.stem.endswith(".local"):
        local = path.with_name(f"{path.stem}.local{path.suffix}")
        if local.exists():
            with local.open(encoding="utf-8") as fh:
                data = _deep_merge(data, yaml.safe_load(fh) or {})

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

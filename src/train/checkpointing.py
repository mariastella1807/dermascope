"""Estado de entrenamiento reanudable.

En Colab la sesion puede desconectarse a mitad de un entrenamiento. Para no empezar de
cero, al final de cada epoca se guarda en disco todo lo necesario para continuar
exactamente donde se quedo:

- pesos del modelo, estado del optimizador, del scheduler y del escalador de AMP;
- epoca completada, mejor metrica, epocas sin mejora e historial;
- estados de los generadores aleatorios (Python, NumPy, PyTorch CPU y GPU), para que el
  barajado de lotes y las aumentaciones de las epocas siguientes sean las mismas que si
  el entrenamiento no se hubiera interrumpido.

Es un archivo distinto del checkpoint del mejor modelo: aquel guarda solo pesos para
evaluar y desplegar; este guarda la ultima epoca para continuar entrenando, y se borra
cuando el entrenamiento termina.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def guardar_estado(ruta: Path, *, model, optimizer, scheduler, scaler, epoch: int,
                   best_metric: float, stale: int, history: list, minutos: float,
                   detenido: bool = False) -> None:
    """Guarda el estado completo al terminar una epoca.

    Se escribe primero en un archivo temporal y luego se renombra: si la sesion se corta
    durante la escritura, el estado anterior sigue intacto en vez de quedar corrupto.
    """
    estado = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "stale": stale,
        "history": history,
        "minutos": minutos,
        "detenido": detenido,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporal = ruta.with_name(ruta.name + ".tmp")
    torch.save(estado, temporal)
    os.replace(temporal, ruta)


def cargar_estado(ruta: Path, *, model, optimizer, scheduler, scaler, device) -> dict | None:
    """Restaura el estado guardado. Devuelve None si no hay nada que reanudar."""
    if not ruta.exists():
        return None
    # weights_only=False: el archivo lo escribe este mismo script y contiene objetos de
    # Python (estados de los generadores aleatorios) ademas de tensores.
    estado = torch.load(ruta, map_location=device, weights_only=False)
    model.load_state_dict(estado["model"])
    optimizer.load_state_dict(estado["optimizer"])
    scheduler.load_state_dict(estado["scheduler"])
    if scaler is not None and estado["scaler"] is not None:
        scaler.load_state_dict(estado["scaler"])

    rng = estado["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    return estado

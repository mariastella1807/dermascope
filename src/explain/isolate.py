"""Aislamiento de la lesion a partir de la mascara (§4.2).

El enunciado exige que, a partir de la mascara, el sistema recorte o aisle el objeto de
interes sobre la imagen original, extrayendolo del fondo, y que ese resultado se muestre
en el aplicativo desplegado.

Postprocesado antes de recortar:

- Cierre morfologico: rellena los huecos que deja la prediccion en lesiones con pelo
  atravesado, donde el modelo suele cortar la mascara a lo largo del vello. Se implementa
  con max pooling (Semana 2): dilatar es tomar el maximo de cada ventana, y erosionar es
  tomar el minimo, que equivale a `-max_pool(-mascara)`. Cerrar es dilatar y luego erosionar.
- Region conectada mas grande: en dermatoscopia hay una sola lesion por imagen, asi que
  cualquier region suelta es ruido de prediccion. Descartarla evita que el recorte
  arrastre trozos de piel sana.
- Area minima: por debajo de un umbral se declara "sin deteccion" en vez de devolver un
  recorte de dos pixeles, que en el app se veria como un fallo silencioso.

La caja del recorte se calcula directamente de los pixeles de la mascara (fila y columna
minima y maxima), igual que las cajas de referencia del notebook de deteccion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.explain.gradcam import resize_map


@dataclass
class IsolationResult:
    """Salida del aislamiento, con todo lo que el app necesita mostrar."""

    mask: np.ndarray          # (H,W) uint8 en 0/1, ya postprocesada
    rgba: np.ndarray          # (H,W,4) lesion sobre fondo transparente
    cropped_rgba: np.ndarray  # recorte a la caja de la lesion
    overlay: np.ndarray       # (H,W,3) original con contorno dibujado
    bbox: tuple[int, int, int, int] | None  # x, y, ancho, alto
    area_ratio: float         # fraccion del area de la imagen ocupada por la lesion
    detected: bool


def _dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)


def _erode(mask: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)


def postprocess_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    keep_largest_component: bool = True,
    morph_close_kernel: int = 5,
    min_area_ratio: float = 0.001,
) -> tuple[np.ndarray, bool]:
    """Convierte el mapa de probabilidad en una mascara binaria limpia."""
    mask = (prob_map >= threshold).astype(np.uint8)

    if morph_close_kernel and morph_close_kernel > 1:
        if morph_close_kernel % 2 == 0:
            raise ValueError("morph_close_kernel debe ser impar para conservar el tamano")
        t = torch.from_numpy(mask).float()[None, None]
        t = _erode(_dilate(t, morph_close_kernel), morph_close_kernel)
        mask = t[0, 0].numpy().astype(np.uint8)

    if keep_largest_component and mask.any():
        # Etiqueta cada region conectada (vecindad de 8) con un entero 1..n.
        labels, n_regions = ndimage.label(mask, structure=np.ones((3, 3)))
        if n_regions > 1:
            sizes = np.bincount(labels.ravel())[1:]  # la etiqueta 0 es el fondo
            mask = (labels == 1 + int(np.argmax(sizes))).astype(np.uint8)

    area_ratio = float(mask.sum()) / float(mask.size)
    return mask, area_ratio >= min_area_ratio


def isolate_lesion(
    image_rgb: np.ndarray,
    prob_map: np.ndarray,
    threshold: float = 0.5,
    keep_largest_component: bool = True,
    morph_close_kernel: int = 5,
    min_area_ratio: float = 0.001,
    contour_color: tuple[int, int, int] = (0, 255, 0),
    pad: int = 8,
) -> IsolationResult:
    """Extrae la lesion del fondo de piel.

    `prob_map` debe estar en [0,1]. Si no tiene el tamano de `image_rgb` (el modelo
    trabaja a 256px), se reescala aqui a la resolucion original.
    """
    image_rgb = np.asarray(image_rgb)
    h, w = image_rgb.shape[:2]
    if prob_map.shape != (h, w):
        prob_map = resize_map(prob_map, h, w)

    mask, detected = postprocess_mask(
        prob_map, threshold, keep_largest_component, morph_close_kernel, min_area_ratio
    )
    area_ratio = float(mask.sum()) / float(mask.size)

    # Lesion sobre fondo transparente: el canal alfa es la mascara.
    rgba = np.dstack([image_rgb, (mask * 255).astype(np.uint8)])

    # Contorno = mascara menos su erosion. Se dilata una vez para que tenga 3 px de grosor.
    t = torch.from_numpy(mask).float()[None, None]
    border = _dilate(t - _erode(t, 3), 3)[0, 0].numpy() > 0
    overlay = image_rgb.copy()
    overlay[border] = contour_color

    bbox = None
    cropped = rgba
    if detected:
        rows, cols = np.nonzero(mask)
        # El margen da contexto perilesional, que es relevante para el diagnostico.
        x0, y0 = max(int(cols.min()) - pad, 0), max(int(rows.min()) - pad, 0)
        x1, y1 = min(int(cols.max()) + 1 + pad, w), min(int(rows.max()) + 1 + pad, h)
        bbox = (x0, y0, x1 - x0, y1 - y0)
        cropped = rgba[y0:y1, x0:x1]

    return IsolationResult(
        mask=mask,
        rgba=rgba,
        cropped_rgba=cropped,
        overlay=overlay,
        bbox=bbox,
        area_ratio=area_ratio,
        detected=detected,
    )

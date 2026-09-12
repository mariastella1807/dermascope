"""Aislamiento de la lesion a partir de la mascara (§4.2).

El enunciado exige que, a partir de la mascara, el sistema recorte o aisle el objeto de
interes sobre la imagen original, extrayendolo del fondo, y que ese resultado se muestre
en el aplicativo desplegado.

Postprocesado antes de recortar:

- Cierre morfologico: rellena los huecos que deja la prediccion en lesiones con pelo
  atravesado, donde el modelo suele cortar la mascara a lo largo del vello.
- Componente conexa mas grande: en dermatoscopia hay una sola lesion por imagen, asi que
  cualquier componente satelite es ruido de prediccion. Descartarla evita que el recorte
  arrastre trozos de piel sana.
- Area minima: por debajo de un umbral se declara "sin deteccion" en vez de devolver un
  recorte de dos pixeles, que en el app se veria como un fallo silencioso.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class IsolationResult:
    """Salida del aislamiento, con todo lo que el app necesita mostrar."""

    mask: np.ndarray          # (H,W) uint8 en 0/1, ya postprocesada
    rgba: np.ndarray          # (H,W,4) lesion sobre fondo transparente
    cropped_rgba: np.ndarray  # recorte al bounding box de la lesion
    overlay: np.ndarray       # (H,W,3) original con contorno dibujado
    bbox: tuple[int, int, int, int] | None  # x, y, w, h
    area_ratio: float         # fraccion del area de la imagen ocupada por la lesion
    detected: bool


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
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if keep_largest_component and mask.any():
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels > 1:
            # La etiqueta 0 es el fondo; se busca el mayor entre las demas.
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)

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

    `prob_map` debe estar en [0,1] y tener el mismo tamano que `image_rgb`; si no, se
    reescala aqui para que el app pueda pasar la salida del modelo sin preprocesarla.
    """
    h, w = image_rgb.shape[:2]
    if prob_map.shape != (h, w):
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    mask, detected = postprocess_mask(
        prob_map, threshold, keep_largest_component, morph_close_kernel, min_area_ratio
    )
    area_ratio = float(mask.sum()) / float(mask.size)

    # Lesion sobre fondo transparente: canal alfa = mascara.
    rgba = np.dstack([image_rgb, (mask * 255).astype(np.uint8)])

    # Contorno sobre la original, para que el usuario vea que se segmento.
    overlay = image_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, contour_color, 2)

    bbox = None
    cropped = rgba
    if detected and contours:
        x, y, bw, bh = cv2.boundingRect(np.vstack(contours))
        # El padding da contexto peri-lesional, que es relevante para el diagnostico.
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
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

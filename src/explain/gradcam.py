"""Grad-CAM y SmoothGrad-CAM implementados desde cero (§4.3).

Se implementa a mano en vez de usar `pytorch-grad-cam` por dos razones: es una
dependencia menos en el despliegue, y en la sustentacion hay que poder explicar el
mecanismo, no invocarlo.

La idea: para la clase c, se toman las activaciones A^k de una capa convolucional y el
gradiente del logit y^c respecto a ellas. El promedio espacial del gradiente,

    alpha_k = mean_ij( dy^c / dA^k_ij )

mide cuanto contribuye el canal k a la clase c. El mapa es la combinacion lineal de
canales ponderada por alpha, pasada por ReLU para conservar solo la evidencia *a favor*
de la clase:

    L^c = ReLU( sum_k alpha_k * A^k )

En este proyecto se aplica sobre `layer4`, que con CBAM activo termina en el bloque de
atencion. Comparar el mapa del modelo con y sin CBAM es la evidencia visual que pide
§4.3: el objetivo es mostrar que con CBAM la masa del mapa se concentra en la lesion y
no en artefactos como pelo, burbujas de inmersion o el vineteado del dermatoscopio.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps


class GradCAM:
    """Grad-CAM sobre una capa objetivo. Usar como context manager o llamar a `close()`.

    Los hooks quedan registrados en el modulo objetivo, asi que hay que liberarlos para
    no acumularlos entre llamadas en una sesion larga de Streamlit.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Devuelve (mapa HxW normalizado a [0,1], indice de clase usado).

        `input_tensor` debe ser un batch de 1: (1, 3, H, W), ya normalizado.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError("Grad-CAM espera un batch de una sola imagen: (1,3,H,W)")

        was_training = self.model.training
        self.model.eval()
        # Se necesita grad aunque estemos en inferencia: el mapa ES un gradiente.
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Los hooks no capturaron nada: la capa objetivo no participo en el forward"
            )

        # alpha_k: promedio espacial del gradiente por canal.
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].cpu().numpy()

        if was_training:
            self.model.train()

        return _normalize(cam), class_idx


class SmoothGradCAM(GradCAM):
    """Promedia Grad-CAM sobre copias de la entrada con ruido gaussiano.

    El mapa crudo de Grad-CAM es sensible al punto exacto de evaluacion del gradiente y
    suele salir ruidoso. Promediar sobre n muestras perturbadas produce un mapa mas
    estable, que es el que conviene mostrar en el aplicativo.
    """

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        n_samples: int = 16,
        noise_std: float = 0.15,
    ) -> tuple[np.ndarray, int]:
        # La clase se fija con la entrada limpia; si no, distintas muestras podrian
        # explicar clases distintas y el promedio no significaria nada.
        if class_idx is None:
            with torch.no_grad():
                class_idx = int(self.model(input_tensor).argmax(dim=1).item())

        sigma = noise_std * float(input_tensor.max() - input_tensor.min())
        accumulator = np.zeros(input_tensor.shape[-2:], dtype=np.float64)
        for _ in range(n_samples):
            noisy = input_tensor + torch.randn_like(input_tensor) * sigma
            cam, _ = super().__call__(noisy, class_idx)
            accumulator += cam

        return _normalize(accumulator / n_samples), class_idx


def _normalize(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    peak = cam.max()
    # Un mapa completamente plano (gradiente nulo) se devuelve en ceros en vez de NaN.
    return (cam / peak).astype(np.float32) if peak > 1e-8 else np.zeros_like(cam, dtype=np.float32)


def resize_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reescala un mapa 2D con interpolacion bilineal."""
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def overlay_heatmap(
    image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Superpone el mapa sobre la imagen original, con la paleta `jet` de matplotlib."""
    image_rgb = np.asarray(image_rgb)
    if cam.shape != image_rgb.shape[:2]:
        cam = resize_map(cam, *image_rgb.shape[:2])
    heat = colormaps["jet"](np.clip(cam, 0, 1))[..., :3] * 255  # (H, W, 4) RGBA -> RGB
    return np.uint8((1 - alpha) * image_rgb + alpha * heat)


def attention_mass_in_mask(cam: np.ndarray, mask: np.ndarray) -> float:
    """Fraccion de la masa del mapa de atencion que cae dentro de la mascara.

    Esta es la metrica cuantitativa para el analisis de §4.3. En vez de afirmar "con
    CBAM el mapa se ve mejor", se reporta un numero: que proporcion de la atencion del
    clasificador cae sobre la lesion segmentada. Si CBAM cumple su funcion, esta
    fraccion sube respecto al modelo sin el bloque.
    """
    if cam.shape != mask.shape:
        cam = resize_map(cam, *mask.shape)
    total = cam.sum()
    if total <= 1e-8:
        return 0.0
    return float(cam[mask > 0.5].sum() / total)

"""Segmentador de la lesion: SegFormer-B0 y baseline U-Net (§4.2, §4.3).

SegFormer aporta el requisito de self-attention de §4.3. Su encoder (MiT) aplica
self-attention eficiente en cuatro escalas, con reduccion espacial de las claves y
valores para que el costo no crezca cuadraticamente con la resolucion. Eso importa aqui
porque una lesion puede ocupar el 5% o el 60% del campo del dermatoscopio, y la atencion
global permite decidir el borde usando contexto de toda la imagen en vez de solo el
vecindario del pixel, que es la limitacion estructural de una U-Net convolucional.

El baseline U-Net existe para poder sostener la comparacion que pide §4.3: mismos datos,
mismas epocas, misma perdida, y la unica diferencia es si el encoder tiene self-attention.

Nota de implementacion: SegFormer emite logits a 1/4 de la resolucion de entrada. Este
wrapper los reescala a tamano completo para que la interfaz de ambos modelos sea la misma
y el resto del codigo no tenga que saber cual esta usando.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerBinary(nn.Module):
    """SegFormer con cabeza de 1 canal y upsample al tamano de entrada."""

    def __init__(self, checkpoint: str = "nvidia/mit-b0", num_classes: int = 1) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig.from_pretrained(checkpoint, num_labels=num_classes)
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

    @property
    def attention_source(self) -> nn.Module:
        """Encoder del que se pueden extraer los pesos de self-attention.

        Llamar al modelo con `output_attentions=True` sobre `self.net.segformer`
        devuelve las matrices de atencion por etapa, que es el material para el
        analisis de mapas de atencion de §4.3.
        """
        return self.net.segformer


class UNetBaseline(nn.Module):
    """U-Net con encoder ResNet preentrenado. Contraste convolucional puro."""

    def __init__(self, encoder: str = "resnet34", num_classes: int = 1) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_segmenter(cfg) -> nn.Module:
    arch = cfg.model.arch
    if arch == "segformer":
        return SegFormerBinary(cfg.model.segformer_checkpoint, cfg.model.num_classes)
    if arch == "unet":
        return UNetBaseline(cfg.model.unet_encoder, cfg.model.num_classes)
    raise ValueError(f"Arquitectura de segmentacion no soportada: {arch}")

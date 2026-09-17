"""Segmentador de la lesion: U-Net con encoder ResNet-34 y self-attention (§4.2, §4.3).

Arquitectura, en terminos de lo visto en clase:

- Encoder: ResNet-34 preentrenada en ImageNet (Semana 3), la misma familia que el
  clasificador. Cada etapa reduce la resolucion a la mitad y entrega un mapa de
  caracteristicas que se guarda para las skip connections.
- Cuello de botella: un bloque de self-attention (`SelfAttention2d`, la operacion de ViT)
  sobre el mapa de 8x8 de `layer4`.
- Decoder: U-Net (Semana 6). Cada bloque sube la resolucion x2, concatena el mapa del
  encoder de la misma escala y aplica dos convoluciones 3x3. Las skip connections
  devuelven el detalle espacial fino que el encoder perdio al reducir, y es lo que
  permite trazar un borde preciso.

Comparacion de §4.3: el mismo modelo se construye con y sin el bloque de atencion. Mismo
encoder, mismo decoder, mismos datos y misma semilla, asi que la diferencia en Dice e
IoU es atribuible al self-attention. Es el mismo diseno de ablacion que CBAM en el
clasificador.

    Imagen 256 -> stem /2 -> layer1 /4 -> layer2 /8 -> layer3 /16 -> layer4 /32 (8x8)
                                                                        |
                                                                 self-attention
                                                                        |
    Mascara 256 <- dec0 /1 <- dec1 /2 <- dec2 /4 <- dec3 /8 <- dec4 /16 <-+
                             (skips: stem, layer1, layer2, layer3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.models.attention import SelfAttention2d

# Etapas del encoder: son las que vienen preentrenadas y reciben el learning rate base.
_ENCODER_PREFIXES = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")


class DecoderBlock(nn.Module):
    """Sube x2, concatena la skip connection y aplica dos Conv3x3-BN-ReLU."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DermaUNet(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        num_classes: int = 1,
        pretrained: bool = True,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()
        if image_size % 32 != 0:
            raise ValueError("image_size debe ser multiplo de 32: el encoder reduce 5 veces x2")

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
        self.use_self_attention = use_self_attention

        # Encoder: las capas de ResNet-34, sin el pooling global ni la capa fc.
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2,  64 canales
        self.pool = resnet.maxpool                                          # /4
        self.layer1 = resnet.layer1                                         # /4,  64
        self.layer2 = resnet.layer2                                         # /8,  128
        self.layer3 = resnet.layer3                                         # /16, 256
        self.layer4 = resnet.layer4                                         # /32, 512

        n_tokens = (image_size // 32) ** 2
        self.attention = (
            SelfAttention2d(512, n_tokens) if use_self_attention else nn.Identity()
        )

        self.dec4 = DecoderBlock(512, 256, 256)  # /16
        self.dec3 = DecoderBlock(256, 128, 128)  # /8
        self.dec2 = DecoderBlock(128, 64, 64)    # /4
        self.dec1 = DecoderBlock(64, 64, 64)     # /2
        self.dec0 = DecoderBlock(64, 0, 32)      # /1, sin skip: ya es la resolucion de entrada
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.attention(self.layer4(x3))

        d = self.dec4(x4, x3)
        d = self.dec3(d, x2)
        d = self.dec2(d, x1)
        d = self.dec1(d, x0)
        d = self.dec0(d)
        return self.head(d)  # logits (B, 1, H, W)

    def param_groups(self, base_lr: float, head_multiplier: float = 10.0):
        """Encoder preentrenado con lr base; atencion y decoder, nuevos, con lr mayor.

        Es el fine-tuning con tasas de aprendizaje diferenciadas de la Semana 3, igual
        que en el clasificador.
        """
        encoder, new = [], []
        for name, param in self.named_parameters():
            (encoder if name.startswith(_ENCODER_PREFIXES) else new).append(param)
        return [
            {"params": encoder, "lr": base_lr},
            {"params": new, "lr": base_lr * head_multiplier},
        ]


def build_segmenter(cfg, override_attention: bool | None = None) -> DermaUNet:
    """Instancia el segmentador desde la configuracion.

    `override_attention` permite la ablacion por linea de comandos (`--no-attention`) sin
    editar el YAML, igual que `--no-cbam` en el clasificador.
    """
    m = cfg.model
    return DermaUNet(
        image_size=cfg.data.image_size,
        num_classes=m.num_classes,
        pretrained=m.pretrained,
        use_self_attention=m.use_self_attention if override_attention is None else override_attention,
    )

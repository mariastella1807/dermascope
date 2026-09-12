"""Bloques de atencion: CBAM y su version reducida SE (§4.3).

CBAM (Woo et al., 2018) refina un mapa de activaciones en dos pasos secuenciales:

    F' = M_c(F) * F          atencion de canal:   que filtros importan
    F'' = M_s(F') * F'       atencion espacial:   donde en la imagen importa

Ambas ramas son multiplicativas y parten de la identidad aproximada, asi que insertarlas
en un backbone preentrenado no destruye los pesos de ImageNet: solo reponderan.

En dermatoscopia la rama espacial es la interesante. La imagen tiene mucho fondo de piel
sana y artefactos (pelo, marcas de regla, vineteado del dermatoscopio); la atencion
espacial le da al modelo un mecanismo explicito para descontarlos.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Rama de canal: MLP compartido sobre avg-pool y max-pool globales.

    Se usan los dos pooling porque aportan informacion distinta: el promedio describe
    la respuesta global del filtro y el maximo captura la evidencia puntual mas fuerte,
    que en una lesion pequena es justamente la senal relevante.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        weights = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    """Rama espacial: convolucion sobre el avg y el max a lo largo de los canales.

    El kernel grande (7x7 por defecto) es intencional: el mapa de atencion debe
    describir la region de la lesion, no bordes finos.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar para conservar la resolucion")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el mapa espacial sin aplicarlo, para visualizarlo en el app."""
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: canal seguido de espacial."""

    def __init__(
        self, channels: int, reduction: int = 16, spatial_kernel: int = 7
    ) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: CBAM sin la rama espacial.

    Incluido como termino intermedio de la ablacion, para poder separar cuanto del
    aporte de CBAM viene del canal y cuanto de la atencion espacial.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel(x)

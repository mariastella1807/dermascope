"""Bloques de atencion (§4.3): CBAM, su version reducida SE, y self-attention.

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

import math

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


class SelfAttention2d(nn.Module):
    """Self-attention sobre un mapa de caracteristicas: la operacion central de ViT.

    Es la formula vista en la Semana 3 con Vision Transformer:

        Atencion(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Lo unico que cambia es de donde salen los tokens. ViT corta la imagen en parches; aqui
    cada posicion del mapa que entrega `layer4` de ResNet-34 es un token. A 256px ese mapa
    es de 8x8, asi que hay 64 tokens y el costo O(N^2) que se discutio en clase es
    pequeno: 64 x 64 = 4.096 productos.

    Por que va en el cuello de botella de la U-Net: una convolucion 3x3 solo ve a sus
    vecinos, y la auto-atencion conecta cada posicion con todas las demas en una sola
    capa. El decodificador recibe asi features que ya "saben" donde esta el resto de la
    lesion, lo que deberia ayudar en lesiones grandes o de borde difuso.

    Tres detalles tomados de ViT y ResNet:
    - Embeddings de posicion aprendibles: la atencion es invariante a permutaciones y sin
      ellos no sabria donde esta cada token.
    - LayerNorm antes de calcular Q, K y V.
    - Conexion residual, x + Atencion(x). La proyeccion de salida arranca en ceros, asi
      que al inicio del entrenamiento el bloque es la identidad y no altera los features
      preentrenados; solo empieza a aportar si eso reduce la perdida.
    """

    def __init__(self, channels: int, n_tokens: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Ultima matriz de atencion (B, N, N), guardada para poder visualizarla.
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C), N = H*W
        if tokens.size(1) != self.pos_embed.size(1):
            raise ValueError(
                f"El bloque espera {self.pos_embed.size(1)} tokens y recibio {tokens.size(1)}: "
                "la imagen de entrada no tiene el tamano de entrenamiento"
            )

        t = self.norm(tokens + self.pos_embed)
        q, k, v = self.query(t), self.key(t), self.value(t)

        scores = q @ k.transpose(1, 2) / math.sqrt(c)  # (B, N, N)
        weights = scores.softmax(dim=-1)                # cada fila suma 1
        self.last_attention = weights.detach()

        tokens = tokens + self.proj(weights @ v)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

"""Pipelines de aumentacion con `torchvision.transforms.v2`.

Criterio de diseno, la pregunta de la Semana 2: ¿la transformacion preserva la etiqueta?

- Volteos: si. Una imagen dermatoscopica no tiene orientacion canonica; un nevus volteado
  sigue siendo un nevus. (Al contrario de los digitos de clase, donde un "2" volteado deja
  de ser un "2".)
- Recorte y escala moderados: si. Cambian el encuadre, no el diagnostico.
- Color: solo con moderacion. El tono y la variacion cromatica son senal diagnostica real
  (la regla ABCD incluye el color), asi que un cambio fuerte de matiz destruye
  informacion en vez de regularizar.

Se usa `transforms.v2` y no la API clasica porque v2 aplica la misma transformacion
geometrica a la imagen y a su mascara en una sola llamada. Sin eso, un volteo aleatorio
podria voltear la imagen y no la mascara, y el modelo aprenderia bordes equivocados.

Todas las pipelines empiezan con `ToImage()`, asi que aceptan tanto una imagen PIL (los
Datasets) como un arreglo NumPy (lo que entrega Streamlit).
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

# Estadisticas de ImageNet: el encoder viene preentrenado con esta normalizacion.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_normalized_tensor() -> list:
    return [
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def classification_train_transform(cfg) -> v2.Compose:
    aug = cfg.augment
    size = cfg.data.image_size
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), ratio=(0.9, 1.11)),
            v2.RandomHorizontalFlip(p=aug.hflip),
            v2.RandomVerticalFlip(p=aug.vflip),
            v2.ColorJitter(
                brightness=aug.brightness,
                contrast=aug.contrast,
                saturation=aug.saturation,
                hue=aug.hue,
            ),
            *_to_normalized_tensor(),
        ]
    )


def classification_eval_transform(cfg) -> v2.Compose:
    size = cfg.data.image_size
    return v2.Compose([v2.ToImage(), v2.Resize((size, size)), *_to_normalized_tensor()])


def segmentation_train_transform(cfg) -> v2.Compose:
    """Se llama como `transform(imagen, tv_tensors.Mask(mascara))`.

    Las transformaciones geometricas (Resize, RandomAffine, volteos) se aplican a las dos;
    las de color y la normalizacion solo a la imagen. v2 usa interpolacion por vecino mas
    cercano en la mascara, asi que sigue siendo binaria despues de rotar o escalar.
    """
    aug = cfg.augment
    size = cfg.data.image_size
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((size, size)),
            v2.RandomAffine(
                degrees=aug.rotate_degrees,
                translate=(aug.shift_limit, aug.shift_limit),
                scale=(1 - aug.scale_limit, 1 + aug.scale_limit),
            ),
            v2.RandomHorizontalFlip(p=aug.hflip),
            v2.RandomVerticalFlip(p=aug.vflip),
            v2.ColorJitter(brightness=aug.brightness, contrast=aug.contrast),
            *_to_normalized_tensor(),
        ]
    )


def segmentation_eval_transform(cfg) -> v2.Compose:
    size = cfg.data.image_size
    return v2.Compose([v2.ToImage(), v2.Resize((size, size)), *_to_normalized_tensor()])

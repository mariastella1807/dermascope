"""Pipelines de aumentacion con Albumentations.

Criterio de diseno: las imagenes dermatoscopicas no tienen orientacion canonica, asi que
todas las simetrias del cuadrado son aumentaciones validas y baratas. En cambio el color
se toca con moderacion, porque el tono y la variacion cromatica son senal diagnostica
real (la regla ABCD incluye el color): saturar o desplazar el matiz agresivamente
destruye informacion en vez de regularizar.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Estadisticas de ImageNet: el backbone viene preentrenado con esta normalizacion.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def classification_train_transform(cfg) -> A.Compose:
    aug = cfg.augment
    size = cfg.data.image_size
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(size, size), scale=(0.7, 1.0), ratio=(0.9, 1.11), p=1.0
            ),
            A.HorizontalFlip(p=aug.hflip),
            A.VerticalFlip(p=aug.vflip),
            A.RandomRotate90(p=aug.rotate90),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=aug.brightness_contrast
            ),
            A.HueSaturationValue(
                hue_shift_limit=aug.hue_shift_limit,
                sat_shift_limit=aug.saturation_shift_limit,
                val_shift_limit=8,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def classification_eval_transform(cfg) -> A.Compose:
    size = cfg.data.image_size
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def segmentation_train_transform(cfg) -> A.Compose:
    aug = cfg.augment
    size = cfg.data.image_size
    return A.Compose(
        [
            A.Resize(size, size),
            A.ShiftScaleRotate(
                shift_limit=aug.shift_limit,
                scale_limit=aug.scale_limit,
                rotate_limit=45,
                border_mode=0,
                p=0.7,
            ),
            A.HorizontalFlip(p=aug.hflip),
            A.VerticalFlip(p=aug.vflip),
            A.RandomRotate90(p=aug.rotate90),
            A.RandomBrightnessContrast(p=aug.brightness_contrast),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def segmentation_eval_transform(cfg) -> A.Compose:
    size = cfg.data.image_size
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

"""Clasificador de 7 clases: ResNet preentrenada con CBAM inyectado (§4.1, §4.3).

Por que ResNet y no EfficientNet: en torchvision, `layer1..layer4` son `nn.Sequential`,
asi que CBAM se puede insertar *dentro* del backbone appendando el bloque al final de
una etapa. Eso es integracion real en la arquitectura, no un modulo pegado detras del
pooling global. Ademas ResNet cuantiza a INT8 sin sorpresas, lo que sostiene §4.4.

La ablacion de §4.3 se obtiene construyendo el mismo modelo con `use_cbam=False`: los
pesos preentrenados, el split y la semilla son identicos, de modo que la diferencia en
macro-F1 es atribuible al bloque de atencion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.models.attention import CBAM

_BACKBONES = {
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class DermaClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet34",
        num_classes: int = 7,
        pretrained: bool = True,
        use_cbam: bool = True,
        cbam_stages: tuple[str, ...] = ("layer3", "layer4"),
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Backbone no soportado: {backbone}")

        ctor, weights = _BACKBONES[backbone]
        self.backbone_name = backbone
        self.use_cbam = use_cbam
        self.net = ctor(weights=weights if pretrained else None)

        # La cabeza se crea ANTES que CBAM. Inicializar CBAM consume numeros aleatorios;
        # si fuera primero, la cabeza arrancaria con pesos distintos en la corrida con y
        # sin CBAM, y la ablacion mezclaria el efecto del bloque con el de la
        # inicializacion. Asi, con la misma semilla, todo lo comun es identico.
        in_features = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        if use_cbam:
            self._inject_cbam(cbam_stages, cbam_reduction, cbam_spatial_kernel)

    def _inject_cbam(
        self, stages: tuple[str, ...], reduction: int, spatial_kernel: int
    ) -> None:
        """Appenda un CBAM al final de cada etapa indicada.

        Solo se instrumentan las etapas profundas (layer3/layer4). Las tempranas
        codifican textura de bajo nivel donde reponderar canales aporta poco y el
        costo de computo sobre mapas grandes es alto.
        """
        for stage_name in stages:
            stage = getattr(self.net, stage_name, None)
            if stage is None:
                raise ValueError(f"La etapa {stage_name} no existe en {self.backbone_name}")
            channels = self._stage_out_channels(stage)
            setattr(
                self.net,
                stage_name,
                nn.Sequential(*stage, CBAM(channels, reduction, spatial_kernel)),
            )

    @staticmethod
    def _stage_out_channels(stage: nn.Module) -> int:
        """Numero de canales de salida de una etapa ResNet.

        Se lee del ultimo BatchNorm del ultimo bloque en vez de hardcodear, para que
        resnet34 (BasicBlock, bn2) y resnet50 (Bottleneck, bn3) funcionen igual.
        """
        last_block = list(stage.children())[-1]
        norms = [m for m in last_block.modules() if isinstance(m, nn.BatchNorm2d)]
        if not norms:
            raise RuntimeError("No se pudo inferir el numero de canales de la etapa")
        return norms[-1].num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Capa objetivo para Grad-CAM: la ultima etapa convolucional.

        Con CBAM activo, esta etapa termina en el bloque de atencion, asi que el mapa
        de Grad-CAM refleja las activaciones ya reponderadas. Eso es justamente lo que
        se quiere comparar contra el modelo sin CBAM.
        """
        return self.net.layer4

    def param_groups(self, base_lr: float, head_multiplier: float = 10.0):
        """Separa backbone y cabeza para aplicar learning rates distintos.

        La cabeza se inicializa de cero y necesita avanzar rapido; el backbone viene de
        ImageNet y un lr alto le borraria las features utiles. Los CBAM se agrupan con
        la cabeza porque tambien parten de inicializacion aleatoria.
        """
        head_params, backbone_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name or "channel" in name or "spatial" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_multiplier},
        ]


def build_classifier(cfg, override_cbam: bool | None = None) -> DermaClassifier:
    """Instancia el clasificador desde la configuracion.

    `override_cbam` permite la ablacion por linea de comandos (`--no-cbam`) sin editar
    el YAML, garantizando que el resto de hiperparametros no cambie.
    """
    m = cfg.model
    return DermaClassifier(
        backbone=m.backbone,
        num_classes=m.num_classes,
        pretrained=m.pretrained,
        use_cbam=m.use_cbam if override_cbam is None else override_cbam,
        cbam_stages=tuple(m.cbam_stages),
        cbam_reduction=m.cbam_reduction,
        cbam_spatial_kernel=m.cbam_spatial_kernel,
        dropout=m.dropout,
    )

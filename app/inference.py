"""Pipeline unificado de inferencia que consume el aplicativo Streamlit.

Se mantiene separado de `streamlit_app.py` a proposito: aqui no hay ninguna llamada a
Streamlit, asi que el pipeline completo se puede probar desde un script o un notebook
sin levantar el servidor. El app solo se encarga de la interfaz.

Orden de ejecucion y por que:

1. Clasificacion sobre la imagen completa. No sobre el recorte: el contexto de piel
   perilesional es informativo y el modelo se entreno con la imagen entera.
2. Grad-CAM sobre la clase predicha, con la misma imagen que se clasifico.
3. Segmentacion, tambien sobre la imagen completa.
4. Aislamiento de la lesion a partir de la mascara (§4.2).
5. Fraccion de la masa de atencion que cae dentro de la mascara predicha: cierra el
   circulo entre las tres tareas y es lo que se muestra como chequeo de coherencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, resolve
from src.data import transforms as T
from src.explain.gradcam import GradCAM, attention_mass_in_mask, overlay_heatmap
from src.explain.isolate import IsolationResult, isolate_lesion
from src.models.classifier import build_classifier
from src.models.segmenter import build_segmenter


@dataclass
class PipelineOutput:
    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float]
    gradcam: np.ndarray          # mapa HxW en [0,1]
    gradcam_overlay: np.ndarray  # imagen con el mapa superpuesto
    segmentation: IsolationResult
    attention_in_lesion: float
    latency_ms: dict[str, float]


class DermaPipeline:
    """Carga ambos modelos una sola vez y ejecuta el pipeline completo."""

    def __init__(
        self,
        cls_config: str = "configs/classification.yaml",
        seg_config: str = "configs/segmentation.yaml",
        cls_checkpoint: str | None = None,
        seg_checkpoint: str | None = None,
        device: str | None = None,
    ) -> None:
        self.cls_cfg = load_config(cls_config)
        self.seg_cfg = load_config(seg_config)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.class_names = list(self.cls_cfg.classes)
        self.class_names_es = dict(self.cls_cfg.class_names_es)

        # Por defecto, los modelos con atencion dentro de paths.models, que los overrides
        # locales pueden mover fuera del repositorio.
        models_dir = resolve(self.cls_cfg.paths.models)
        self.classifier = self._load(
            build_classifier(self.cls_cfg),
            cls_checkpoint or models_dir / "classifier_cbam_best.pt",
        )
        self.segmenter = self._load(
            build_segmenter(self.seg_cfg),
            seg_checkpoint or models_dir / "segmenter_attn_best.pt",
        )

        self.cls_tf = T.classification_eval_transform(self.cls_cfg)
        self.seg_tf = T.segmentation_eval_transform(self.seg_cfg)

    def _load(self, model: torch.nn.Module, checkpoint: str | Path) -> torch.nn.Module:
        path = resolve(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"Falta el checkpoint {path}. Entrena el modelo o copialo desde Colab."
            )
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        return model.to(self.device).eval()

    @torch.no_grad()
    def _classify(self, image_rgb: np.ndarray) -> tuple[int, np.ndarray, torch.Tensor]:
        tensor = self.cls_tf(image_rgb).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.classifier(tensor), dim=1)[0].cpu().numpy()
        return int(probs.argmax()), probs, tensor

    @torch.no_grad()
    def _segment(self, image_rgb: np.ndarray) -> np.ndarray:
        tensor = self.seg_tf(image_rgb).unsqueeze(0).to(self.device)
        logits = self.segmenter(tensor)
        return torch.sigmoid(logits)[0, 0].cpu().numpy()

    def run(self, image_rgb: np.ndarray) -> PipelineOutput:
        import time

        latency: dict[str, float] = {}

        start = time.perf_counter()
        class_idx, probs, cls_tensor = self._classify(image_rgb)
        latency["clasificacion"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        with GradCAM(self.classifier, self.classifier.gradcam_target_layer) as cam_fn:
            cam, _ = cam_fn(cls_tensor, class_idx)
        latency["gradcam"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        prob_map = self._segment(image_rgb)
        latency["segmentacion"] = (time.perf_counter() - start) * 1000

        post = self.seg_cfg.postprocess
        isolation = isolate_lesion(
            image_rgb,
            prob_map,
            threshold=post.threshold,
            keep_largest_component=post.keep_largest_component,
            morph_close_kernel=post.morph_close_kernel,
            min_area_ratio=post.min_area_ratio,
        )

        return PipelineOutput(
            predicted_class=self.class_names[class_idx],
            predicted_index=class_idx,
            probabilities={
                name: float(p) for name, p in zip(self.class_names, probs)
            },
            gradcam=cam,
            gradcam_overlay=overlay_heatmap(image_rgb, cam),
            segmentation=isolation,
            attention_in_lesion=attention_mass_in_mask(
                cam, isolation.mask.astype(np.float32)
            ),
            latency_ms=latency,
        )

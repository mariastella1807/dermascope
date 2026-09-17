"""Cuantizacion INT8 estatica del clasificador con ONNX Runtime (§4.4).

El enunciado pide un proceso explicito de optimizacion con comparacion de metricas *y*
de tamano del modelo antes y despues. Este script produce las tres cosas:

1. Exporta el clasificador entrenado a ONNX FP32.
2. Cuantiza a INT8 estatico usando un lector de calibracion sobre imagenes reales del
   split de entrenamiento.
3. Evalua accuracy y macro-F1 de ambas versiones sobre el mismo split de test y reporta
   los tamanos en disco.

Por que INT8 estatico y no dinamico: la cuantizacion dinamica solo afecta capas lineales,
y ResNet-34 es casi toda convolucion, asi que la reduccion seria marginal. La estatica
cuantiza tambien los pesos y activaciones de las convoluciones, y por eso necesita el
paso de calibracion: hay que observar el rango real de las activaciones con datos
representativos para elegir las escalas.

Advertencia importante para el informe: el modelo INT8 corre en CPU. La comparacion de
§4.5 GPU vs CPU se hace con el modelo FP32; la de §4.4 se hace CPU-FP32 contra CPU-INT8,
sobre el mismo motor (ONNX Runtime), porque medir PyTorch-FP32 contra ORT-INT8 mezclaria
el efecto del runtime con el de la cuantizacion.

Uso:
    python -m src.optimize.quantize --config configs/classification.yaml \
        --checkpoint models/classifier_cbam_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, resolve, set_seed


def onnx_size_mb(path: Path) -> float:
    """Tamano en disco del modelo ONNX, sumando sus archivos de pesos externos si los hay.

    Algunos exportadores guardan la estructura en `modelo.onnx` y los pesos en
    `modelo.onnx.data`. Medir solo el primero reportaria un modelo de pocos KB y la
    comparacion de tamano antes y despues seria falsa.
    """
    return sum(f.stat().st_size for f in path.parent.glob(path.name + "*")) / 1e6


def export_onnx(model: torch.nn.Module, path: Path, image_size: int, opset: int) -> Path:
    """Exporta a ONNX con batch dinamico, para poder medir batch 1 y 8 sin reexportar."""
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # Exportador clasico: genera un unico archivo con los pesos incluidos y un grafo
        # que el preprocesado de ONNX Runtime analiza sin problemas. El exportador nuevo
        # (dynamo) separa los pesos en otro archivo y su grafo hace fallar la inferencia
        # de formas previa a la cuantizacion.
        dynamo=False,
    )
    print(f"ONNX FP32 exportado: {path} ({onnx_size_mb(path):.2f} MB)")
    return path


class CalibrationReader:
    """Alimenta al cuantizador con imagenes reales del split de entrenamiento.

    Usar ruido gaussiano aqui es el error clasico: las escalas quedarian calibradas para
    una distribucion de activaciones que nunca ocurre en produccion, y el macro-F1 se
    desploma en las clases minoritarias.
    """

    def __init__(self, dataset, n_samples: int, input_name: str = "input") -> None:
        from onnxruntime.quantization import CalibrationDataReader  # noqa: F401

        self.input_name = input_name
        indices = np.random.default_rng(0).choice(
            len(dataset), size=min(n_samples, len(dataset)), replace=False
        )
        self._batches = iter(
            [
                {input_name: dataset[int(i)][0].unsqueeze(0).numpy().astype(np.float32)}
                for i in indices
            ]
        )

    def get_next(self):
        return next(self._batches, None)

    def rewind(self) -> None:
        raise NotImplementedError("Se recalibra creando un lector nuevo")


def quantize_static(fp32_path: Path, int8_path: Path, reader) -> Path:
    from onnxruntime.quantization import QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # El preprocesado infiere shapes y hace folding; sin este paso, quantize_static
    # deja muchas convoluciones en FP32 y la reduccion de tamano es mucho menor.
    prepared = fp32_path.with_name(fp32_path.stem + "_prep.onnx")
    quant_pre_process(str(fp32_path), str(prepared))

    quantize_static(
        model_input=str(prepared),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    prepared.unlink(missing_ok=True)
    print(f"ONNX INT8 generado: {int8_path} ({onnx_size_mb(int8_path):.2f} MB)")
    return int8_path


def evaluate_onnx(onnx_path: Path, loader, class_names: list[str]) -> dict:
    """Accuracy y macro-F1 del modelo ONNX sobre un DataLoader."""
    import onnxruntime as ort

    from src.eval.metrics import classification_metrics

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    y_true, y_pred = [], []
    for images, labels in loader:
        logits = session.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        y_pred.extend(logits.argmax(axis=1).tolist())
        y_true.extend(labels.tolist())

    return classification_metrics(np.array(y_true), np.array(y_pred), class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Por defecto, classifier_cbam_best.pt dentro de paths.models",
    )
    args = parser.parse_args()

    from src.data import transforms as T
    from src.data.datasets import DermaClassificationDataset, make_dataloaders
    from src.models.classifier import build_classifier

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    models_dir = resolve(cfg.paths.models)
    class_names = list(cfg.classes)

    # El checkpoint vive en paths.models, que los overrides locales mueven fuera del
    # repo (OneDrive en local, /content en Colab). Anclarlo a la raiz del repo hacia
    # que el default no existiera en ninguna de las dos maquinas.
    checkpoint = resolve(args.checkpoint) if args.checkpoint else models_dir / "classifier_cbam_best.pt"
    if not checkpoint.exists():
        raise SystemExit(
            f"No existe {checkpoint}. Entrena el clasificador con CBAM antes de cuantizar."
        )

    model = build_classifier(cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)

    loaders, datasets = make_dataloaders(cfg, "classification")

    fp32 = export_onnx(
        model, models_dir / "classifier_fp32.onnx", cfg.data.image_size, cfg.optimize.onnx_opset
    )
    # La calibracion usa imagenes de train, pero con la transformacion de EVALUACION.
    # Las de entrenamiento (recortes, volteos, cambios de color aleatorios) producirian
    # rangos de activacion que no ocurren al evaluar ni en el aplicativo, y las escalas
    # INT8 quedarian ajustadas a una distribucion distinta de la real.
    calibration_set = DermaClassificationDataset(
        datasets["train"].frame, T.classification_eval_transform(cfg)
    )
    reader = CalibrationReader(calibration_set, cfg.optimize.calibration_samples)
    int8 = quantize_static(fp32, models_dir / "classifier_int8.onnx", reader)

    print("\nEvaluando FP32 ...")
    metrics_fp32 = evaluate_onnx(fp32, loaders["test"], class_names)
    print("Evaluando INT8 ...")
    metrics_int8 = evaluate_onnx(int8, loaders["test"], class_names)

    f1_drop = metrics_fp32["macro_f1"] - metrics_int8["macro_f1"]
    summary = {
        "fp32": {
            "size_mb": onnx_size_mb(fp32),
            "accuracy": metrics_fp32["accuracy"],
            "macro_f1": metrics_fp32["macro_f1"],
            "per_class_f1": metrics_fp32["per_class_f1"],
        },
        "int8": {
            "size_mb": onnx_size_mb(int8),
            "accuracy": metrics_int8["accuracy"],
            "macro_f1": metrics_int8["macro_f1"],
            "per_class_f1": metrics_int8["per_class_f1"],
        },
        "size_reduction": 1 - (onnx_size_mb(int8) / onnx_size_mb(fp32)),
        "macro_f1_drop": f1_drop,
        "within_tolerance": bool(f1_drop <= cfg.optimize.max_f1_drop),
    }

    out = resolve(cfg.paths.reports) / "results" / "optimization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== §4.4 Optimizacion ===")
    print(f"Tamano   FP32 {summary['fp32']['size_mb']:.2f} MB -> INT8 {summary['int8']['size_mb']:.2f} MB "
          f"(-{summary['size_reduction'] * 100:.1f}%)")
    print(f"Accuracy FP32 {summary['fp32']['accuracy']:.4f} -> INT8 {summary['int8']['accuracy']:.4f}")
    print(f"MacroF1  FP32 {summary['fp32']['macro_f1']:.4f} -> INT8 {summary['int8']['macro_f1']:.4f} "
          f"(caida {f1_drop:.4f})")
    if not summary["within_tolerance"]:
        print(
            f"\nLa caida supera max_f1_drop={cfg.optimize.max_f1_drop}. "
            "Revisar las clases minoritarias en per_class_f1 y considerar "
            "mas muestras de calibracion o la alternativa de poda (src/optimize/prune.py)."
        )
    print(f"\nDetalle en {out}")


if __name__ == "__main__":
    main()

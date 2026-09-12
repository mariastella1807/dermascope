"""Medicion de tiempos de inferencia en CPU y GPU (§4.5).

El enunciado pide una comparacion clara que incluya el hardware especifico de cada
medicion, asi que el script registra automaticamente CPU, GPU, versiones y numero de
hilos junto a cada fila. Las mediciones se corren en maquinas distintas (CPU local, GPU
en Colab) y se consolidan en el mismo CSV por acumulacion.

Tres detalles que hacen la diferencia entre un numero util y uno inventado:

- **Warmup.** Las primeras pasadas incluyen la carga de kernels de cuDNN y la
  inicializacion del contexto CUDA. Sin warmup, la GPU parece mas lenta que la CPU.
- **Sincronizacion.** Las llamadas CUDA son asincronas; sin `torch.cuda.synchronize()`
  se mide el tiempo de encolar el trabajo, no el de ejecutarlo.
- **Percentiles.** Se reporta la mediana y el p95 ademas de la media, porque la media
  sola esconde la variabilidad que el usuario percibe en el app.

Uso:
    python -m src.eval.benchmark --device cpu
    python -m src.eval.benchmark --device cuda --batch-sizes 1 8
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import REPO_ROOT, load_config, resolve


def hardware_info(device: str) -> dict:
    """Descripcion del hardware que se anota junto a cada medicion."""
    info = {
        "device": device,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "gpu": "",
        "cuda": "",
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se pidio --device cuda pero CUDA no esta disponible")
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda or ""
    return info


@torch.no_grad()
def time_model(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    device: str,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict:
    """Latencia por batch en milisegundos."""
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    for _ in range(n_warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(timings)
    batch = input_shape[0]
    return {
        "batch_size": batch,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "ms_per_image": float(arr.mean() / batch),
        "fps": float(1000.0 * batch / arr.mean()),
        "n_runs": n_runs,
    }


def onnx_latency(
    onnx_path: Path, input_shape: tuple[int, ...], n_warmup: int = 10, n_runs: int = 50
) -> dict:
    """Latencia del modelo cuantizado INT8 en ONNX Runtime (solo CPU).

    Se mide aparte porque el modelo INT8 no corre en el runtime de PyTorch: la
    comparacion honesta de §4.4 es FP32 en ONNX Runtime contra INT8 en ONNX Runtime,
    sobre el mismo motor de ejecucion.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(n_warmup):
        session.run(None, {input_name: x})

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        timings.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(timings)
    batch = input_shape[0]
    return {
        "batch_size": batch,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "ms_per_image": float(arr.mean() / batch),
        "fps": float(1000.0 * batch / arr.mean()),
        "n_runs": n_runs,
    }


def append_results(rows: list[dict], out_path: Path) -> None:
    """Acumula en el CSV en vez de sobrescribir: CPU y GPU se miden en maquinas distintas."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if out_path.exists():
        frame = pd.concat([pd.read_csv(out_path), frame], ignore_index=True)
    frame.to_csv(out_path, index=False)
    print(f"\nResultados acumulados en {out_path}")
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cls-config", default="configs/classification.yaml")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument(
        "--include-onnx",
        action="store_true",
        help="Mide tambien el INT8 exportado (solo tiene sentido en CPU)",
    )
    args = parser.parse_args()

    from src.models.classifier import build_classifier
    from src.models.segmenter import build_segmenter

    cls_cfg = load_config(args.cls_config)
    seg_cfg = load_config(args.seg_config)
    hw = hardware_info(args.device)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows: list[dict] = []
    targets = [
        ("classifier_cbam", build_classifier(cls_cfg, override_cbam=True), cls_cfg.data.image_size),
        ("classifier_nocbam", build_classifier(cls_cfg, override_cbam=False), cls_cfg.data.image_size),
        ("segmenter", build_segmenter(seg_cfg), seg_cfg.data.image_size),
    ]

    for name, model, size in targets:
        for batch in args.batch_sizes:
            print(f"Midiendo {name} en {args.device}, batch={batch} ...")
            result = time_model(
                model, (batch, 3, size, size), args.device, n_runs=args.n_runs
            )
            rows.append({"model": name, "precision": "fp32", "timestamp": timestamp, **hw, **result})

    if args.include_onnx:
        models_dir = resolve(cls_cfg.paths.models)
        for name, filename, precision in [
            ("classifier_cbam", "classifier_fp32.onnx", "fp32-onnx"),
            ("classifier_cbam", "classifier_int8.onnx", "int8-onnx"),
        ]:
            path = models_dir / filename
            if not path.exists():
                print(f"Omitido {filename}: no existe. Corre src.optimize.quantize primero.")
                continue
            for batch in args.batch_sizes:
                print(f"Midiendo {filename}, batch={batch} ...")
                result = onnx_latency(path, (batch, 3, cls_cfg.data.image_size, cls_cfg.data.image_size), n_runs=args.n_runs)
                rows.append({"model": name, "precision": precision, "timestamp": timestamp, **hw, **result})

    out = resolve(cls_cfg.paths.reports) / "results" / "timings.csv"
    append_results(rows, out)

    # Copia en JSON con el entorno completo, para trazabilidad en el informe.
    env_path = out.parent / f"env_{args.device}.json"
    env_path.write_text(json.dumps(hw, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

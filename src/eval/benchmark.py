"""Medicion de tiempos de inferencia en CPU y GPU (§4.5).

El enunciado pide una comparacion clara que incluya el hardware especifico de cada
medicion, asi que el script registra automaticamente el procesador (nombre comercial),
la GPU, las versiones y el numero de hilos junto a cada fila. Las mediciones se corren en
maquinas distintas (CPU local, GPU en Colab) y se consolidan en el mismo CSV por
acumulacion.

Cuatro detalles que hacen la diferencia entre un numero util y uno inventado:

- **Warmup.** Las primeras pasadas incluyen la carga de kernels de cuDNN y la
  inicializacion del contexto CUDA. Sin warmup, la GPU parece mas lenta que la CPU.
- **Sincronizacion.** Las llamadas CUDA son asincronas; sin `torch.cuda.synchronize()`
  se mide el tiempo de encolar el trabajo, no el de ejecutarlo.
- **Medicion intercalada.** En un portatil la CPU se calienta y baja su frecuencia a los
  pocos minutos. Si los modelos se midieran uno despues de otro, el ultimo quedaria en
  desventaja. Por eso cada ronda ejecuta una pasada de cada modelo, en un orden sorteado
  en cada ronda: todos comparten las mismas condiciones termicas y ninguno se mide
  siempre justo despues del mismo modelo pesado. PyTorch y ONNX Runtime se intercalan
  en grupos separados: los hilos de un motor que quedan esperando trabajo le quitan
  nucleos al otro, y mezclarlos hacia parecer mas lento al modelo medido justo despues.
- **Percentiles.** Se reporta la mediana y el p95 ademas de la media, porque la media
  sola esconde la variabilidad que el usuario percibe en el app.

Uso:
    python -m src.eval.benchmark --device cpu --include-onnx
    python -m src.eval.benchmark --device cuda --batch-sizes 1 8
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.config import load_config, resolve


def cpu_name() -> str:
    """Nombre comercial del procesador, p. ej. '13th Gen Intel(R) Core(TM) i3-1305U'.

    `platform.processor()` en Windows solo devuelve la familia ("Intel64 Family 6 Model
    186"), que no permite saber con que equipo se midio.
    """
    try:
        if platform.system() == "Windows":
            import winreg

            clave = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, clave) as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if platform.system() == "Linux":
            for linea in Path("/proc/cpuinfo").read_text().splitlines():
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def power_source() -> str:
    """'corriente' o 'bateria' en Windows: con bateria el sistema baja la frecuencia de la CPU."""
    if platform.system() != "Windows":
        return ""
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    estado = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(estado)):
        return ""
    return {0: "bateria", 1: "corriente"}.get(estado.ACLineStatus, "desconocido")


def hardware_info(device: str) -> dict:
    """Descripcion del hardware que se anota junto a cada medicion."""
    info = {
        "device": device,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "power": power_source(),
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


def summarize(timings_ms: list[float], batch: int) -> dict:
    arr = np.array(timings_ms)
    return {
        "batch_size": batch,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "ms_per_image": float(arr.mean() / batch),
        "fps": float(1000.0 * batch / arr.mean()),
        "n_runs": len(arr),
    }


@torch.no_grad()
def measure_interleaved(
    targets: list[tuple[tuple, Callable[[], object]]],
    device: str,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> dict[tuple, list[float]]:
    """Latencia de cada objetivo, en ms, medida por rondas intercaladas.

    `targets` es una lista de (clave, funcion sin argumentos que ejecuta una inferencia).
    El orden se sortea en cada ronda (con semilla fija, para que sea reproducible). Rotar
    el punto de partida no basta: conserva los vecinos, y un modelo que siempre se mide
    despues del mas pesado hereda una CPU mas caliente o cachés desplazadas.
    """
    for _, run in targets:
        for _ in range(n_warmup):
            run()
    if device == "cuda":
        torch.cuda.synchronize()

    timings: dict[tuple, list[float]] = {key: [] for key, _ in targets}
    generador = np.random.default_rng(0)
    for _ in range(n_runs):
        for i in generador.permutation(len(targets)):
            key, run = targets[i]
            start = time.perf_counter()
            run()
            if device == "cuda":
                torch.cuda.synchronize()
            timings[key].append((time.perf_counter() - start) * 1000.0)
    return timings


def torch_runner(model: torch.nn.Module, input_shape: tuple[int, ...], device: str) -> Callable[[], object]:
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    return lambda: model(x)


def onnx_runner(onnx_path: Path, input_shape: tuple[int, ...]) -> Callable[[], object]:
    """Inferencia en ONNX Runtime (solo CPU).

    El modelo INT8 no corre en el runtime de PyTorch: la comparacion honesta de §4.4 es
    FP32 en ONNX Runtime contra INT8 en ONNX Runtime, sobre el mismo motor de ejecucion.
    """
    import onnxruntime as ort

    opciones = ort.SessionOptions()
    # Mismos hilos que PyTorch, y sin espera activa: por defecto los hilos de ONNX Runtime
    # siguen ocupando nucleos un rato despues de cada inferencia.
    opciones.intra_op_num_threads = torch.get_num_threads()
    opciones.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(onnx_path), opciones, providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: np.random.randn(*input_shape).astype(np.float32)}
    return lambda: session.run(None, feed)


def append_results(rows: list[dict], out_path: Path) -> None:
    """Acumula en el CSV en vez de sobrescribir: CPU y GPU se miden en maquinas distintas."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if out_path.exists():
        frame = pd.concat([pd.read_csv(out_path), frame], ignore_index=True)
    frame.to_csv(out_path, index=False)
    print(f"\nResultados acumulados en {out_path}")
    print(pd.DataFrame(rows)[["model", "precision", "batch_size", "p50_ms", "p95_ms", "ms_per_image", "fps"]]
          .round(2).to_string(index=False))


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
        help="Mide tambien el clasificador exportado a ONNX, FP32 e INT8 (solo CPU)",
    )
    args = parser.parse_args()

    from src.models.classifier import build_classifier
    from src.models.segmenter import build_segmenter

    cls_cfg = load_config(args.cls_config)
    seg_cfg = load_config(args.seg_config)
    hw = hardware_info(args.device)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Hardware: {hw['cpu']} | {hw['gpu'] or 'sin GPU'} | hilos {hw['threads']} | {hw['power'] or ''}")

    # Los pesos no influyen en el tiempo: se construyen los modelos sin cargar checkpoints.
    modelos = [
        ("classifier_cbam", build_classifier(cls_cfg, override_cbam=True), cls_cfg.data.image_size),
        ("classifier_nocbam", build_classifier(cls_cfg, override_cbam=False), cls_cfg.data.image_size),
        ("segmenter_attn", build_segmenter(seg_cfg, override_attention=True), seg_cfg.data.image_size),
        ("segmenter_noattn", build_segmenter(seg_cfg, override_attention=False), seg_cfg.data.image_size),
    ]
    grupo_torch = []
    for name, model, size in modelos:
        for batch in args.batch_sizes:
            grupo_torch.append(((name, "fp32", batch), torch_runner(model, (batch, 3, size, size), args.device)))
    grupos = [grupo_torch]

    if args.include_onnx:
        if args.device != "cpu":
            raise SystemExit("--include-onnx mide ONNX Runtime en CPU: usalo con --device cpu")
        models_dir = resolve(cls_cfg.paths.models)
        size = cls_cfg.data.image_size
        grupo_onnx = []
        for filename, precision in [("classifier_fp32.onnx", "fp32-onnx"), ("classifier_int8.onnx", "int8-onnx")]:
            path = models_dir / filename
            if not path.exists():
                print(f"Omitido {filename}: no existe. Corre src.optimize.quantize primero.")
                continue
            for batch in args.batch_sizes:
                grupo_onnx.append((("classifier_cbam", precision, batch), onnx_runner(path, (batch, 3, size, size))))
        grupos.append(grupo_onnx)

    timings: dict[tuple, list[float]] = {}
    for grupo in grupos:
        print(f"Midiendo {len(grupo)} combinaciones de modelo y lote, {args.n_runs} rondas intercaladas ...")
        timings.update(measure_interleaved(grupo, args.device, n_runs=args.n_runs))
    rows = [
        {"model": name, "precision": precision, "timestamp": timestamp, **hw, **summarize(t, batch)}
        for (name, precision, batch), t in timings.items()
    ]

    out = resolve(cls_cfg.paths.reports) / "results" / "timings.csv"
    append_results(rows, out)

    # Copia en JSON con el entorno completo, para trazabilidad en el informe.
    env_path = out.parent / f"env_{args.device}.json"
    env_path.write_text(json.dumps(hw, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

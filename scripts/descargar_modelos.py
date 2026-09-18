"""Descarga los modelos entrenados desde el release `modelos-v1` del repositorio.

Los checkpoints no se versionan en git (pesan entre 20 y 100 MB). Este script los deja en
la carpeta que indica `paths.models` en `configs/paths.yaml` (por defecto `models/`), y
verifica la huella SHA-256 de cada archivo, así que una descarga cortada no pasa
inadvertida. Los archivos que ya están y coinciden no se vuelven a descargar.

    python scripts/descargar_modelos.py            # los 2 modelos que usa el aplicativo
    python scripts/descargar_modelos.py --todos    # los 6: también las versiones sin atención y ONNX
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, resolve  # noqa: E402

URL_RELEASE = "https://github.com/mariastella1807/dermascope/releases/download/modelos-v1"

# archivo: huella SHA-256 publicada en el release
MODELOS = {
    "classifier_cbam_best.pt": "b665138727d40fab5d5fd93e9de177033d83712f6f45c215d962d5d481df4e10",
    "segmenter_attn_best.pt": "786a154de8a4e8537743ddbd368ad603885be0b71b83e2a745109f080df27224",
    "classifier_nocbam_best.pt": "e7489d2b8a63d8e9bd25823c1dfea6bce929af4f8e17a237bbd92ee8352b6e22",
    "segmenter_noattn_best.pt": "3fba8a4ff913c211dd88be4a22b2e165b210aefac91c1f31f95926526b71ed68",
    "classifier_fp32.onnx": "be652220a351739f542fe942aa55555e371119d6d2e70afa8801d29e57195574",
    "classifier_int8.onnx": "95af0db2c20c694ecf21bb6fca8715c0eb6a30e0f84ee9010702c81fcb4df972",
}
MODELOS_APP = ["classifier_cbam_best.pt", "segmenter_attn_best.pt"]


def huella_sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--todos", action="store_true", help="descarga los 6 modelos, no solo los del aplicativo")
    args = parser.parse_args()

    destino_dir = resolve(load_config(REPO_ROOT / "configs" / "paths.yaml").paths.models)
    destino_dir.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta de modelos: {destino_dir}")

    for nombre in (MODELOS if args.todos else MODELOS_APP):
        destino = destino_dir / nombre
        if destino.exists() and huella_sha256(destino) == MODELOS[nombre]:
            print(f"  ya está y es correcto: {nombre}")
            continue
        parcial = destino.with_name(nombre + ".part")
        print(f"  descargando {nombre} ...", flush=True)
        urllib.request.urlretrieve(f"{URL_RELEASE}/{nombre}", parcial)
        if huella_sha256(parcial) != MODELOS[nombre]:
            parcial.unlink()
            print(f"  ERROR: {nombre} llegó incompleto o alterado. Vuelve a ejecutar el script.")
            return 1
        parcial.replace(destino)
        print(f"  verificado: {nombre} ({destino.stat().st_size / 1e6:.1f} MB)")

    print("Listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

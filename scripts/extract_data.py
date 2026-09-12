"""Extrae los zips de HAM10000 de forma reanudable y con memoria acotada.

`Expand-Archive` y `ZipFile.ExtractToDirectory` de .NET fallan en dos puntos en esta
maquina: no se pueden reanudar si el proceso muere a mitad, y abortan cuando el destino
ya contiene alguno de los archivos. Con 10.015 imagenes y una maquina de 7,7 GB de RAM
donde el watchdog del sistema mata procesos, las dos cosas importan.

Este script copia entrada por entrada en bloques de 1 MB, salta lo que ya existe con el
tamano correcto y se puede volver a lanzar tantas veces como haga falta.

Uso:
    python scripts/extract_data.py <destino> <zip> [<zip> ...]

Ejemplo:
    python scripts/extract_data.py C:\\ml-data\\dermascope\\raw\\HAM10000_images `
        C:\\ml-data\\dermascope\\raw\\HAM10000_images_part_1.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

CHUNK = 1024 * 1024  # 1 MB: suficiente para ir rapido sin retener el archivo completo


def extract(zip_path: Path, dest: Path, report_every: int = 500) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    stats = {"extraidos": 0, "omitidos": 0, "carpetas": 0}

    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.infolist()
        total = len(entries)
        print(f"{zip_path.name}: {total} entradas -> {dest}")

        for i, info in enumerate(entries, start=1):
            name = info.filename
            # Las entradas de macOS y los directorios no aportan nada.
            if info.is_dir() or name.startswith("__MACOSX") or "/." in name:
                stats["carpetas"] += 1
                continue

            target = dest / Path(name).name  # aplana: los zips traen todo en la raiz
            if target.exists() and target.stat().st_size == info.file_size:
                stats["omitidos"] += 1
            else:
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                stats["extraidos"] += 1

            if i % report_every == 0 or i == total:
                print(
                    f"  {i}/{total}  extraidos={stats['extraidos']} "
                    f"omitidos={stats['omitidos']}",
                    flush=True,
                )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path, help="Carpeta destino")
    parser.add_argument("zips", type=Path, nargs="+", help="Archivos .zip a extraer")
    args = parser.parse_args()

    missing = [z for z in args.zips if not z.exists()]
    if missing:
        print(f"No existen: {', '.join(str(m) for m in missing)}", file=sys.stderr)
        return 1

    grand = {"extraidos": 0, "omitidos": 0, "carpetas": 0}
    for zip_path in args.zips:
        stats = extract(zip_path, args.dest)
        for key, value in stats.items():
            grand[key] += value

    final = len(list(args.dest.glob("*.jpg"))) or len(list(args.dest.glob("*.png")))
    print(f"\nTotal: {grand}")
    print(f"Archivos en {args.dest}: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Obtención y trazabilidad del dataset

El enunciado (§3) exige documentar claramente la fuente y el proceso de conformación del
dataset. Este archivo es ese registro.

## Fuentes

| Componente | Fuente primaria | Licencia | Qué aporta |
| --- | --- | --- | --- |
| Imágenes + etiquetas de 7 clases | HAM10000, Harvard Dataverse (ViDIR Group, Universidad Médica de Viena) — también distribuido como ISIC 2018 Task 3 | CC BY-NC 4.0 | 10.015 imágenes dermatoscópicas con diagnóstico y `lesion_id` |
| Máscaras de segmentación | ISIC 2018 Challenge, Task 1 (Lesion Boundary Segmentation) | CC BY-NC 4.0 | 2.594 máscaras binarias de entrenamiento |

Ambos conjuntos provienen del archivo ISIC y comparten el identificador de imagen
`ISIC_xxxxxxx`, lo que permite cruzarlos sin anotación manual. Esa es la razón de haber
elegido este dominio: las dos tareas obligatorias del proyecto quedan cubiertas por
anotaciones oficiales existentes.

**Uso no comercial.** La licencia CC BY-NC permite el uso académico de este proyecto,
pero obliga a citar la fuente y prohíbe el uso comercial. Citar en el informe:

> Tschandl, P., Rosendahl, C., Kittler, H. (2018). The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin lesions.
> *Scientific Data*, 5, 180161.

> Codella, N. et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018:
> A Challenge Hosted by the International Skin Imaging Collaboration (ISIC).
> arXiv:1902.03368.

## Descarga

Las dos rutas funcionan; Kaggle suele ser más rápida y no requiere registro en el portal
del challenge.

### Opción A — Kaggle (recomendada)

El CLI de Kaggle (2.2.x) ya viene instalado en el `.venv`. La autenticación tiene dos
caminos; el primero es el más simple porque no hay token que administrar:

```powershell
.\.venv\Scripts\kaggle.exe auth login          # flujo OAuth en el navegador
```

Si se prefiere token manual: Kaggle → Settings → API → *Generate New Token*, y guardarlo
en `%USERPROFILE%\.kaggle\access_token` o exportarlo como `KAGGLE_API_TOKEN`.

Descarga (la ruta de destino sale de `configs/paths.local.yaml`):

```powershell
.\.venv\Scripts\kaggle.exe datasets download -d kmader/skin-cancer-mnist-ham10000 `
    -p C:\ml-data\dermascope\raw --unzip
```

Las máscaras del Task 1 **no** están en ese dataset. Descargarlas del portal oficial del
challenge (opción B) o de un espejo verificable en Kaggle, y anotar en el informe cuál
se usó.

### Opción B — Portales oficiales

1. **HAM10000** — Harvard Dataverse, DOI `10.7910/DVN/DBW86T`.
   Descomprimir `HAM10000_images_part_1.zip` y `HAM10000_images_part_2.zip`.
2. **Máscaras** — `challenge.isic-archive.com`, ISIC 2018 Task 1,
   archivo `ISIC2018_Task1_Training_GroundTruth.zip`.

## Estructura esperada

`src/data/build_splits.py` busca estas rutas. En este equipo el dataset vive **fuera de
OneDrive**, en `C:\ml-data\dermascope`, definido en `configs/paths.local.yaml` (archivo
no versionado). La razón: son ~3 GB de JPEG que OneDrive sincronizaría, y además puede
bloquear archivos justo mientras el DataLoader los lee. Cada integrante crea su propio
`paths.local.yaml`; si no existe, se usan las rutas relativas `data/raw` de
`configs/paths.yaml`.

```
C:\ml-data\dermascope\raw\
├── HAM10000_metadata.csv              # image_id, lesion_id, dx, dx_type, age, sex, localization
├── HAM10000_images/                   # ISIC_0024306.jpg, ...
│   ├── HAM10000_images_part_1/        # también acepta las dos subcarpetas originales
│   └── HAM10000_images_part_2/
└── ISIC2018_Task1_masks/              # ISIC_0024306_segmentation.png, ...
```

## Conformación de los splits

```powershell
python -m src.data.build_splits --config configs/paths.yaml
```

El script genera `data/processed/splits.csv` y aplica dos decisiones que hay que poder
defender en la sustentación:

1. **Agrupamiento por `lesion_id`.** HAM10000 contiene varias fotografías de la misma
   lesión física (~7.470 lesiones para 10.015 imágenes). Repartir por imagen pone la
   misma lesión en train y en test, y el modelo la reconoce en vez de generalizar. El
   script aborta con `AssertionError` si alguna lesión aparece en más de un split.
2. **Estratificación por diagnóstico a nivel de lesión.** Con `df` en ~115 imágenes y
   `vasc` en ~142, un reparto aleatorio puede dejar una clase sin representación en
   validación o test.

El CSV marca con `has_mask` las filas que tienen máscara. El dataset de segmentación
filtra por esa columna, así que **ambas tareas heredan el mismo split**: ninguna lesión
que el clasificador vio en entrenamiento aparece en el test del segmentador ni al revés.

## Verificación

Al terminar, el script imprime la tabla de imágenes por split y clase, el número de
lesiones únicas por split y la cobertura de máscaras. Guardar esa salida: es el material
de la sección de dataset del informe.

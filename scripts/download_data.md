# Obtención y trazabilidad del dataset

El enunciado (§3) exige documentar claramente la fuente y el proceso de conformación del
dataset. Este archivo es ese registro.

## Fuente

**Todo proviene de un único depósito: Harvard Dataverse, DOI `10.7910/DVN/DBW86T`**
(ViDIR Group, Universidad Médica de Viena). No requiere cuenta ni autenticación: la API
de Dataverse permite descarga anónima.

| Archivo | Tamaño | Contenido |
| --- | --- | --- |
| `HAM10000_metadata.tab` | 0,8 MB | `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`, `dataset` |
| `HAM10000_images_part_1.zip` | 1,30 GB | ~5.000 imágenes dermatoscópicas |
| `HAM10000_images_part_2.zip` | 1,34 GB | ~5.000 imágenes dermatoscópicas |
| `HAM10000_segmentations_lesion_tschandl.zip` | 10,3 MB | 10.015 máscaras binarias de lesión |

**Licencia:** CC BY-NC 4.0. Uso académico con atribución, sin uso comercial.

### Por qué esta fuente y no el portal del ISIC

El plan inicial era combinar HAM10000 (clasificación) con las máscaras del ISIC 2018
Task 1 (segmentación), que son **2.594**. Al consultar el Dataverse apareció
`HAM10000_segmentations_lesion_tschandl.zip`, que contiene máscaras para **las 10.015
imágenes** del dataset completo, con la misma convención de nombre
`ISIC_xxxxxxx_segmentation.png`.

Tres ventajas sobre el plan original:

1. El conjunto de segmentación pasa de 2.594 a ~10.015 imágenes, casi 4× más datos.
2. Ambas tareas operan sobre **exactamente el mismo conjunto de imágenes**, así que el
   split por `lesion_id` queda alineado entre las dos sin subconjuntos parciales.
3. Una sola fuente y una sola licencia que documentar y citar, en vez de dos.

Estas máscaras fueron generadas de forma semiautomática y revisadas manualmente por el
grupo de Tschandl. Conviene declararlo en el informe: no son anotación experta
píxel a píxel independiente, y eso acota la interpretación del Dice alcanzable.

### Citas obligatorias

> Tschandl, P., Rosendahl, C., Kittler, H. (2018). The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin lesions.
> *Scientific Data*, 5, 180161.

> Tschandl, P. (2020). *HAM10000 dataset*. Harvard Dataverse.
> https://doi.org/10.7910/DVN/DBW86T

## Descarga

No hace falta Kaggle ni registro en challenge.isic-archive.com. Los identificadores
numéricos son los `id` de archivo que devuelve la API de Dataverse.

```powershell
$raw = "C:\ml-data\dermascope\raw"
New-Item -ItemType Directory -Force $raw | Out-Null
$ProgressPreference = 'SilentlyContinue'   # sin esto Invoke-WebRequest va muy lento

# Metadata y mascaras (rapido)
Invoke-WebRequest "https://dataverse.harvard.edu/api/access/datafile/4338392" -OutFile "$raw\HAM10000_metadata.tab"
Invoke-WebRequest "https://dataverse.harvard.edu/api/access/datafile/3838943" -OutFile "$raw\segmentations.zip"

# Imagenes (2,6 GB, lento)
Invoke-WebRequest "https://dataverse.harvard.edu/api/access/datafile/3172585" -OutFile "$raw\HAM10000_images_part_1.zip"
Invoke-WebRequest "https://dataverse.harvard.edu/api/access/datafile/3172584" -OutFile "$raw\HAM10000_images_part_2.zip"
```

Para listar los archivos y sus `id` actuales:

```powershell
Invoke-RestMethod "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T"
```

Descomprimir las máscaras en `HAM10000_segmentations` y las dos partes de imágenes en
`HAM10000_images`. El zip de máscaras trae una carpeta `__MACOSX` que se puede descartar.

## Ubicación de los datos

En este equipo el dataset vive **fuera de OneDrive**, en `C:\ml-data\dermascope`,
definido en `configs/paths.local.yaml` (archivo no versionado). La razón: son ~3 GB que
OneDrive sincronizaría, y además puede bloquear archivos justo mientras el DataLoader
los está leyendo. Cada integrante crea su propio `paths.local.yaml`; si no existe, se
usan las rutas relativas `data/raw` de `configs/paths.yaml`.

## Estructura esperada

```
C:\ml-data\dermascope\raw\
├── HAM10000_metadata.tab           # build_splits detecta el separador por la extension
├── HAM10000_images\                # ISIC_0024306.jpg, ...
└── HAM10000_segmentations\         # ISIC_0024306_segmentation.png, ...
```

## Conformación de los splits

```powershell
.\.venv\Scripts\python.exe -m src.data.build_splits --config configs/paths.yaml
```

El script genera `splits.csv` y aplica dos decisiones que hay que poder defender en la
sustentación:

1. **Agrupamiento por `lesion_id`.** HAM10000 contiene varias fotografías de la misma
   lesión física (~7.470 lesiones para 10.015 imágenes). Repartir por imagen pone la
   misma lesión en train y en test, y el modelo la reconoce en vez de generalizar. El
   script aborta con `AssertionError` si alguna lesión aparece en más de un split.
2. **Estratificación por diagnóstico a nivel de lesión.** Con `df` en ~115 imágenes y
   `vasc` en ~142, un reparto aleatorio puede dejar una clase sin representación en
   validación o test.

El CSV marca con `has_mask` las filas que tienen máscara. Como la cobertura ahora es
prácticamente total, ambas tareas comparten el mismo split y ninguna lesión que el
clasificador vio en entrenamiento aparece en el test del segmentador ni al revés.

## Verificación

Al terminar, el script imprime la tabla de imágenes por split y clase, el número de
lesiones únicas por split y la cobertura de máscaras. Guardar esa salida: es el material
de la sección de dataset del informe.

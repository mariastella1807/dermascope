# DermaScope — Sistema integrado de clasificación, segmentación y atención

Proyecto 2 de Visión Computacional con Deep Learning
Maestría en Inteligencia Artificial y Ciencia de Datos — Universidad Autónoma de Occidente

**Integrantes:** Maria Stella Fuentes Diaz · Sergio Luis Castaño Rodríguez · Alejandro Galvez Cardenas · Joan Sebastian Mena Ortega

> **Propuesta en revisión.** Para revisar el proyecto (dónde está cada cosa, cómo ver los
> resultados sin ejecutar nada, cómo correr los notebooks en Colab y cómo usar la app en tu
> computador) empieza por **[`docs/GUIA_REVISION.md`](docs/GUIA_REVISION.md)**.
>
> | Para | Ir a |
> | --- | --- |
> | Ver los resultados, sin ejecutar nada | [Notebook 01 ejecutado](https://nbviewer.org/github/mariastella1807/dermascope/blob/main/notebooks/ejecutados/01_entrenar_clasificador_colab.ipynb) · [Notebook 02 ejecutado](https://nbviewer.org/github/mariastella1807/dermascope/blob/main/notebooks/ejecutados/02_entrenar_segmentador_colab.ipynb) |
> | Ejecutarlos en Colab, con los modelos ya entrenados | [Abrir 01 en Colab](https://colab.research.google.com/github/mariastella1807/dermascope/blob/main/notebooks/01_entrenar_clasificador_colab.ipynb) · [Abrir 02 en Colab](https://colab.research.google.com/github/mariastella1807/dermascope/blob/main/notebooks/02_entrenar_segmentador_colab.ipynb) |
> | Leer los entregables | [Model card](MODEL_CARD.md) · [Documento breve](reports/informe.md) |
> | Descargar los modelos entrenados | [Release `modelos-v1`](https://github.com/mariastella1807/dermascope/releases/tag/modelos-v1) |
> | Usar la app | [Guía, sección 6](docs/GUIA_REVISION.md#6-usar-la-app-en-tu-computador) |

## Problema

Sistema de apoyo al triaje dermatoscópico. Dada una imagen dermatoscópica, el sistema:

1. **Clasifica** la lesión en una de 7 categorías diagnósticas.
2. **Segmenta** la frontera de la lesión.
3. **Aísla** la lesión del fondo de piel a partir de la máscara.
4. **Explica** la predicción mediante mapas de atención (CBAM + Grad-CAM).

El sistema es una herramienta de apoyo educativo y de priorización, **no un dispositivo
diagnóstico**. Ver `MODEL_CARD.md` para alcances, limitaciones y consideraciones de sesgo.

## Dataset

Una sola fuente: **Harvard Dataverse, DOI `10.7910/DVN/DBW86T`** (descarga anónima, sin
cuenta). Ambas anotaciones cubren el mismo conjunto de imágenes y se cruzan por el
identificador `ISIC_xxxxxxx`:

| Anotación | Archivo | Volumen |
| --- | --- | --- |
| Clasificación, 7 clases | `HAM10000_metadata.tab` | 10.015 imágenes |
| Máscaras binarias de lesión | `HAM10000_segmentations_lesion_tschandl.zip` | 10.015 máscaras |

Las máscaras de Tschandl cubren el dataset completo, no solo las 2.594 del ISIC 2018
Task 1 que se contemplaban inicialmente. Fueron generadas de forma semiautomática y
revisadas manualmente, lo que conviene declarar en el informe: acotan el Dice alcanzable.

Clases: `akiec` (queratosis actínica / carcinoma intraepitelial), `bcc` (carcinoma
basocelular), `bkl` (lesiones benignas tipo queratosis), `df` (dermatofibroma),
`mel` (melanoma), `nv` (nevus melanocítico), `vasc` (lesiones vasculares).

### Dos propiedades del dataset que condicionan el diseño

1. **Múltiples imágenes por lesión física.** HAM10000 tiene ~7.470 `lesion_id` únicos para
   10.015 imágenes. Un split por imagen pone la misma lesión en train y test e infla las
   métricas. `src/data/build_splits.py` agrupa por `lesion_id` antes de estratificar.
2. **Desbalance severo.** `nv` ≈ 6.705 imágenes vs `df` ≈ 115. De ahí que se reporte
   macro-F1 y no solo accuracy, y que la pérdida sea ponderada por clase.

Ver `scripts/download_data.md` para el procedimiento de descarga y la trazabilidad de la fuente.

## Arquitecturas

| Tarea | Modelo | Justificación |
| --- | --- | --- |
| Clasificación | ResNet-34 (ImageNet) + CBAM | `layer1..4` son `nn.Sequential`, así que CBAM se integra *dentro* del backbone; cuantiza limpio en INT8 |
| Segmentación | U-Net con encoder ResNet-34 (ImageNet) + self-attention | Decoder con skip connections para bordes precisos; el bloque de self-attention (operación del Transformer) aporta contexto global (§4.3) |

El proyecto usa un conjunto reducido de dependencias: PyTorch, `torchvision` (modelos y
`transforms.v2`), scikit-learn, scipy, matplotlib, ONNX y Streamlit.
`scripts/smoke_test.py` verifica que no se cuele ninguna otra librería.

## Mecanismos de atención (§4.3)

Cada mecanismo tiene una comparación propia, no una sola ablación compartida:

- **CBAM** — inyectado tras `layer3` y `layer4` del clasificador. Ablación con/sin bloque,
  mismo split y misma semilla. Se compara accuracy, macro-F1 y tamaño del modelo.
- **Self-attention** — bloque `softmax(QKᵀ/√dₖ)V` en el cuello de botella de la U-Net
  (8x8 = 64 tokens a 256px). Ablación con/sin bloque, mismo split y misma semilla. Se
  compara Dice e IoU.
- **Grad-CAM** — implementado en `src/explain/gradcam.py` (sin dependencias externas, para
  poder explicarlo en la sustentación). Se contrastan los mapas del modelo con y sin CBAM.

## Estructura del repositorio

```
configs/          Hiperparámetros y rutas (YAML, sin valores mágicos en el código)
data/raw/         HAM10000 + máscaras de Tschandl (no versionado)
data/processed/   Splits generados, agrupados por lesion_id
src/data/         Construcción de splits, Datasets y transforms
src/models/       CBAM, clasificador, segmentador
src/train/        Loops de entrenamiento
src/eval/         Métricas, evaluación y benchmark CPU vs GPU (§4.5)
src/explain/      Grad-CAM y aislamiento del objeto (§4.2)
src/optimize/     Cuantización INT8 y poda (§4.4)
app/              Aplicativo Streamlit (§4.6)
scripts/          Setup del entorno, descarga de datos, Cloudflare Tunnel
reports/          Tablas de métricas, timings y figuras para el informe
models/           Checkpoints (no versionado)
```

## Puesta en marcha

PyTorch no publica ruedas estables para Python 3.14, que es el intérprete por defecto de
esta máquina. El entorno usa **Python 3.11 o 3.12**.

```powershell
.\scripts\setup_env.ps1        # crea .venv con py -3.12 e instala requirements
.\.venv\Scripts\python.exe scripts\smoke_test.py   # 11 comprobaciones, no necesita el dataset
```

### Configuración local por integrante

`load_config` aplica `configs/<nombre>.local.yaml` sobre el archivo pedido si existe, con
fusión profunda. Esos archivos están en `.gitignore`, así que cada integrante ajusta sus
rutas y su presupuesto de memoria sin generar conflictos, y **Colab usa los valores
versionados** porque los overrides locales no viajan al repositorio.

| Override | Para qué |
| --- | --- |
| `paths.local.yaml` | Dataset y checkpoints fuera de OneDrive |
| `classification.local.yaml` | Menos workers en máquinas con poca RAM |
| `segmentation.local.yaml` | Batch y workers reducidos en local |

Correr `scripts/smoke_test.py` tras cualquier cambio de dependencias: detecta cambios de
API antes de que aparezcan a mitad de un entrenamiento en Colab.

Luego, en orden:

```powershell
python -m src.data.build_splits --config configs/paths.yaml
python -m src.train.train_classifier --config configs/classification.yaml
python -m src.train.train_classifier --config configs/classification.yaml --no-cbam   # ablación
python -m src.train.train_segmenter  --config configs/segmentation.yaml
python -m src.train.train_segmenter  --config configs/segmentation.yaml --no-attention  # ablación
python -m src.optimize.quantize      --config configs/classification.yaml
python -m src.eval.benchmark         --device cpu
python -m src.eval.benchmark         --device cuda
streamlit run app/streamlit_app.py
.\scripts\run_tunnel.ps1
```

El entrenamiento está pensado para ejecutarse en Colab/Kaggle (T4). El benchmark de §4.5 se
corre dos veces: CPU en esta máquina, GPU en Colab, y ambos resultados se consolidan en
`reports/results/timings.csv` con el hardware anotado.

## Entregables (§5)

| Entregable | Ubicación |
| --- | --- |
| Repositorio con entrenamiento, optimización y app | este repo |
| Model card | `MODEL_CARD.md` |
| Documento de problemática, arquitectura y optimización | `reports/informe.md` |
| Tablas CPU vs GPU | `reports/results/timings.csv` |
| App en vivo por Cloudflare Tunnel | `scripts/run_tunnel.ps1` |

# Model Card — DermaScope

> Las celdas marcadas `PENDIENTE` se llenan con la salida de
> `reports/results/*.json` una vez entrenados los modelos. No dejar ninguna sin llenar
> antes de la entrega: §4.7 exige métricas de desempeño en la model card.

**Versión:** 0.1 (estructura) · **Fecha:** PENDIENTE · **Equipo:** PENDIENTE
**Institución:** Universidad Autónoma de Occidente — Maestría en IA y Ciencia de Datos

## 1. Descripción del problema

Sistema de apoyo al triaje dermatoscópico. Dada una imagen dermatoscópica de una lesión
cutánea, el sistema clasifica la lesión en 7 categorías diagnósticas, segmenta su
frontera, la aísla del fondo de piel y explica la predicción mediante mapas de atención.

**Uso previsto.** Herramienta académica y educativa. Apoyo a la priorización visual en
contextos de formación.

**Uso NO previsto.** Diagnóstico clínico, decisiones de tratamiento, tamizaje poblacional,
o cualquier uso sobre pacientes reales. El sistema no está validado clínicamente ni
tiene aprobación regulatoria de ningún tipo.

## 2. Dataset

| Componente | Fuente | Volumen | Licencia |
| --- | --- | --- | --- |
| Imágenes + clases | HAM10000 (ISIC 2018 Task 3) | 10.015 imágenes, 7.470 lesiones | CC BY-NC 4.0 |
| Máscaras | ISIC 2018 Task 1 | 2.594 máscaras binarias | CC BY-NC 4.0 |

Clases: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

**Distribución de clases (desbalance severo).**

| Clase | Aproximado | Fracción |
| --- | --- | --- |
| nv (nevus) | 6.705 | 67% |
| mel (melanoma) | 1.113 | 11% |
| bkl | 1.099 | 11% |
| bcc | 514 | 5% |
| akiec | 327 | 3% |
| vasc | 142 | 1,4% |
| df | 115 | 1,1% |

Verificar estos conteos contra la salida real de `src/data/build_splits.py` y corregir.

**Partición.** 70/15/15 agrupada por `lesion_id` y estratificada por diagnóstico. El
agrupamiento es obligatorio porque HAM10000 contiene varias fotografías de la misma
lesión física; un reparto por imagen filtra información al conjunto de prueba.

**Procedencia.** Imágenes de dos centros: el Departamento de Dermatología de la
Universidad Médica de Viena (Austria) y una práctica de cáncer de piel en Queensland
(Australia). Esta procedencia determina las limitaciones de la sección 6.

Detalle completo del proceso en `scripts/download_data.md`.

## 3. Arquitecturas

| Tarea | Modelo | Preentrenamiento | Parámetros |
| --- | --- | --- | --- |
| Clasificación (§4.1) | ResNet-34 + CBAM en `layer3` y `layer4` | ImageNet-1k | PENDIENTE |
| Segmentación (§4.2) | SegFormer-B0 (`nvidia/mit-b0`) | ImageNet-1k | PENDIENTE |
| Baseline segmentación | U-Net, encoder ResNet-34 | ImageNet-1k | PENDIENTE |

**Mecanismos de atención (§4.3).**

- **CBAM** (atención de canal + espacial) integrado en el backbone del clasificador.
- **Self-attention** aportado por el encoder MiT de SegFormer en el segmentador.
- **SmoothGrad-CAM** como herramienta de interpretabilidad, mostrada en el aplicativo.

## 4. Métricas de desempeño

Todas medidas en el conjunto de prueba, agrupado por `lesion_id`.

### 4.1 Clasificación — ablación de CBAM

| Modelo | Accuracy | Balanced acc. | Macro-F1 | Params |
| --- | --- | --- | --- | --- |
| ResNet-34 sin CBAM | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |
| ResNet-34 + CBAM | PENDIENTE | PENDIENTE | PENDIENTE | PENDIENTE |

F1 por clase: PENDIENTE — copiar de `reports/results/ablacion_cbam.csv`.

El macro-F1 es la métrica principal, no la accuracy: con `nv` al 67% del dataset, un
clasificador constante alcanza ~67% de accuracy sin haber aprendido nada.

### 4.2 Segmentación — aporte del self-attention

| Modelo | Dice | IoU | Params |
| --- | --- | --- | --- |
| U-Net (convolucional) | PENDIENTE | PENDIENTE | PENDIENTE |
| SegFormer-B0 (self-attention) | PENDIENTE | PENDIENTE | PENDIENTE |

### 4.3 Localización de la atención

| Modelo | Masa de Grad-CAM dentro de la lesión |
| --- | --- |
| Sin CBAM | PENDIENTE |
| Con CBAM | PENDIENTE |

### 4.4 Optimización

| Versión | Tamaño | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| FP32 (ONNX) | PENDIENTE | PENDIENTE | PENDIENTE |
| INT8 estático (ONNX) | PENDIENTE | PENDIENTE | PENDIENTE |

### 4.5 Tiempos de inferencia

| Modelo | Dispositivo | Hardware | Batch | ms/imagen | FPS |
| --- | --- | --- | --- | --- | --- |
| Clasificador | CPU | PENDIENTE | 1 | PENDIENTE | PENDIENTE |
| Clasificador | GPU | PENDIENTE | 1 | PENDIENTE | PENDIENTE |
| Segmentador | CPU | PENDIENTE | 1 | PENDIENTE | PENDIENTE |
| Segmentador | GPU | PENDIENTE | 1 | PENDIENTE | PENDIENTE |

Fuente: `reports/results/timings.csv`.

## 5. Alcances

Condiciones en las que el sistema se comporta como fue evaluado:

- Imágenes **dermatoscópicas** (con dermatoscopio de contacto o inmersión), no
  fotografías clínicas ni de teléfono. El dominio de entrenamiento es exclusivamente
  dermatoscópico y el cambio de modalidad degrada el desempeño de forma severa.
- Lesión **única y centrada** en el campo de visión. El postprocesado conserva solo la
  componente conexa más grande, así que una imagen con dos lesiones reportará una.
- Las 7 categorías del dataset. Cualquier condición fuera de esas clases se asignará
  forzosamente a la más parecida, con confianza potencialmente alta.
- Iluminación y encuadre comparables a los de HAM10000.

## 6. Limitaciones

- **Clases minoritarias.** `df` y `vasc` tienen ~115 y ~142 imágenes. Su F1 es el menos
  confiable del sistema y su intervalo de confianza real es amplio: el test contiene
  apenas unas decenas de ejemplos de cada una.
- **Melanoma.** Es la clase con mayor consecuencia clínica de un falso negativo y no es
  la mejor clasificada. Reportar explícitamente su recall, no solo el F1 agregado.
- **Confianza mal calibrada.** El softmax de una red entrenada con pérdida ponderada no
  es una probabilidad calibrada. Las cifras del aplicativo son puntajes relativos, no
  probabilidades de enfermedad.
- **Sin estimación de incertidumbre.** El sistema no sabe cuándo no sabe: ante una
  imagen fuera de distribución emite una predicción con confianza aparente.
- **Artefactos.** Vello denso, burbujas de inmersión, marcas de regla y el viñeteado del
  dermatoscopio afectan tanto la segmentación como la atención. La métrica de masa de
  atención dentro de la lesión, visible en el aplicativo, sirve como señal de alerta.
- **No evaluado:** lesiones en mucosas, uñas y cuero cabelludo; piel pediátrica;
  lesiones post-tratamiento o biopsiadas; imágenes de baja resolución o desenfocadas.

## 7. Consideraciones éticas y de sesgo

- **Sesgo de fototipo cutáneo.** Esta es la limitación más importante del sistema.
  HAM10000 proviene de poblaciones austríaca y australiana, predominantemente de
  fototipos claros (Fitzpatrick I–III). Los fototipos IV–VI están sustancialmente
  subrepresentados, y la presentación dermatoscópica de las lesiones difiere con la
  pigmentación de base. **El desempeño reportado no es transferible a población
  colombiana sin revalidación**, y en particular puede ser sistemáticamente peor en
  pacientes de piel más oscura — la misma población en la que el melanoma se diagnostica
  más tarde y con peor pronóstico. El dataset no incluye anotación de fototipo, así que
  este sesgo **no se puede cuantificar** con los datos disponibles: se declara, no se mide.
- **Sesgo de verificación.** No todos los diagnósticos de HAM10000 están confirmados por
  histopatología; parte proviene de seguimiento clínico o consenso de expertos. Las
  etiquetas de las clases benignas son menos firmes que las de las malignas.
- **Riesgo de falsa tranquilidad.** El daño principal de un sistema así no es un error
  aislado sino que alguien postergue una consulta por un resultado "benigno". De ahí la
  advertencia permanente en la interfaz.
- **Distribución de errores.** Un falso negativo de melanoma y un falso positivo de
  nevus no son equivalentes en consecuencia clínica, aunque la accuracy los cuente igual.
  El informe debe discutir la matriz de confusión, no solo los agregados.
- **Datos.** Licencia CC BY-NC: uso académico con atribución, sin uso comercial. Las
  imágenes están anonimizadas en la fuente; el aplicativo no almacena las imágenes
  cargadas por los usuarios.

## 8. Reproducibilidad

| Elemento | Valor |
| --- | --- |
| Semilla | 42 (`configs/paths.yaml`) |
| Entorno | Python 3.12, ver `requirements.txt` |
| Hardware de entrenamiento | PENDIENTE |
| Comandos | ver `README.md` |

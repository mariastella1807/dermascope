# Model Card — DermaScope

**Versión:** 1.0 · **Fecha:** 18 de septiembre de 2026
**Integrantes:** Maria Stella Fuentes Diaz · Sergio Luis Castaño Rodríguez · Alejandro Galvez Cardenas · Joan Sebastian Mena Ortega
**Institución:** Universidad Autónoma de Occidente — Maestría en Inteligencia Artificial y Ciencia de Datos
**Repositorio:** https://github.com/mariastella1807/dermascope

## 1. Descripción del problema

La dermatoscopia es una técnica de imagen no invasiva que, con aumento óptico e
iluminación controlada, hace visibles estructuras pigmentadas de la piel que no se ven a
simple vista. Su interpretación exige entrenamiento y varía entre observadores.

DermaScope recibe una imagen dermatoscópica de una lesión cutánea y:

1. **clasifica** la lesión en 7 categorías diagnósticas, con la probabilidad de cada una;
2. **segmenta** la lesión, es decir, marca qué píxeles pertenecen a ella;
3. **aísla** la lesión del fondo de piel a partir de la máscara;
4. **explica** la clasificación con un mapa de Grad-CAM y mide qué parte de ese mapa cae
   dentro de la lesión.

**Uso previsto.** Herramienta académica y educativa para estudiar clasificación,
segmentación y mecanismos de atención sobre imágenes dermatoscópicas.

**Uso no previsto.** Diagnóstico clínico, decisiones de tratamiento, tamizaje poblacional
o cualquier uso sobre pacientes reales. El sistema no está validado clínicamente ni tiene
aprobación regulatoria.

## 2. Dataset

**HAM10000** (Tschandl, Rosendahl y Kittler, 2018), conjunto de entrenamiento de la Tarea 3
del challenge ISIC 2018. Harvard Dataverse, DOI `10.7910/DVN/DBW86T`, licencia CC BY-NC 4.0.

| Componente | Archivo | Volumen |
| --- | --- | --- |
| Imágenes y diagnósticos | `HAM10000_metadata.tab` + `HAM10000_images_part_{1,2}.zip` | 10.015 imágenes de 600×450 px, 7.470 lesiones |
| Máscaras de la lesión | `HAM10000_segmentations_lesion_tschandl.zip` | 10.015 máscaras binarias (cobertura del 100%) |

**Procedencia.** Dos centros: el Departamento de Dermatología de la Universidad Médica de
Viena (Austria) y una consulta especializada en cáncer de piel en Queensland (Australia).

**Diagnóstico de referencia.** Histopatología en el 53,3% de las imágenes; el resto, por
seguimiento clínico (37,0%), consenso de expertos (9,0%) o microscopía confocal (0,7%).
En las clases malignas o premalignas (`mel`, `bcc`, `akiec`) la confirmación
histopatológica es del 100%.

**Máscaras.** Generadas de forma semiautomática y revisadas manualmente; no son una
anotación experta píxel a píxel. Esto acota el Dice alcanzable: parte del error medido
es variabilidad de la propia anotación.

**Distribución de clases y partición.** La partición 70/15/15 se hace por `lesion_id`,
estratificada por diagnóstico y con semilla 42: una misma lesión aparece en varias
fotografías (1.956 lesiones tienen más de una), y repartirlas por imagen pondría fotos de
la misma lesión en entrenamiento y en test. Se verificó que ninguna lesión queda en dos
particiones.

| Clase | Diagnóstico | Total | % | Train | Val | Test |
| --- | --- | --- | --- | --- | --- | --- |
| `nv` | Nevus melanocítico | 6.705 | 66,9 | 4.667 | 1.018 | 1.020 |
| `mel` | Melanoma | 1.113 | 11,1 | 798 | 155 | 160 |
| `bkl` | Lesión benigna tipo queratosis | 1.099 | 11,0 | 771 | 162 | 166 |
| `bcc` | Carcinoma basocelular | 514 | 5,1 | 375 | 74 | 65 |
| `akiec` | Queratosis actínica / carcinoma intraepitelial | 327 | 3,3 | 225 | 51 | 51 |
| `vasc` | Lesión vascular | 142 | 1,4 | 101 | 20 | 21 |
| `df` | Dermatofibroma | 115 | 1,1 | 77 | 19 | 19 |
| | **Total** | **10.015** | | **7.014** | **1.499** | **1.502** |

## 3. Arquitecturas

| Tarea | Modelo | Preentrenamiento | Parámetros |
| --- | --- | --- | --- |
| Clasificación | ResNet-34 con CBAM al final de `layer3` y `layer4` | ImageNet-1k | 21,33 M |
| Segmentación | U-Net con encoder ResNet-34 y bloque de self-attention en el cuello de botella | ImageNet-1k (encoder) | 25,61 M |

**Mecanismos de atención.**

- **CBAM** (Woo et al., 2018): atención de canal y espacial integrada dentro del backbone
  del clasificador. Añade 41.156 parámetros (+0,19%).
- **Self-attention** (`softmax(QKᵀ/√d)V`, la operación del Transformer y de ViT) sobre los
  64 tokens del mapa de 8×8 que entrega el encoder, con embeddings de posición, LayerNorm
  y conexión residual. Añade 1,08 M de parámetros (+4,4%). Los puntajes `QKᵀ` se calculan
  en float32: en float16 desbordaban y producían NaN durante el entrenamiento.
- **Grad-CAM** (Selvaraju et al., 2017) sobre `layer4`, como herramienta de
  interpretabilidad en los notebooks y en el aplicativo.

**Entrenamiento.** AdamW con tasas de aprendizaje diferenciadas (×10 para las capas
nuevas), `ReduceLROnPlateau` sobre la pérdida de validación, precisión mixta y 25 épocas
completas sin parada temprana; se conserva el checkpoint de la mejor época en validación.
El clasificador usa entropía cruzada ponderada por la frecuencia inversa de cada clase y
se selecciona por macro-F1; el segmentador usa BCE + Dice y se selecciona por Dice.

## 4. Métricas de desempeño

Todas las cifras son del conjunto de **test** (1.502 imágenes, lesiones que ningún modelo
vio al entrenar). Cada comparación con y sin atención cambia un solo factor: misma semilla,
partición, aumentaciones, hiperparámetros y número de épocas. Para cada diferencia se
reporta un intervalo de confianza del 95% por bootstrap pareado (10.000 remuestreos del
test), que mide la variación por la composición del test pero **no** la variación entre
entrenamientos con distintas semillas.

### 4.1 Clasificación y efecto de CBAM

| Modelo | Accuracy | Accuracy balanceada | Macro-F1 |
| --- | --- | --- | --- |
| ResNet-34 sin CBAM | 0,725 | 0,726 | 0,638 |
| **ResNet-34 + CBAM** (modelo final) | 0,720 | 0,746 | **0,650** |
| Diferencia (IC 95%) | −0,005 (−0,025 a 0,014) | +0,020 (−0,009 a 0,051) | +0,012 (−0,019 a 0,043) |

Ninguna de las diferencias globales es significativa. La métrica principal es el
macro-F1: con `nv` en el 67% del test, un clasificador que siempre responde "nevus"
alcanza 0,679 de accuracy pero solo 0,116 de macro-F1.

**Por clase, modelo final (con CBAM):**

| Clase | Imágenes de test | Precisión | Recall | F1 | F1 sin CBAM |
| --- | --- | --- | --- | --- | --- |
| `akiec` | 51 | 0,49 | 0,69 | 0,57 | 0,51 |
| `bcc` | 65 | 0,60 | 0,86 | 0,70 | 0,74 |
| `bkl` | 166 | 0,64 | 0,63 | 0,64 | 0,59 |
| `df` | 19 | 0,44 | 0,79 | 0,57 | 0,60 |
| `mel` | 160 | 0,33 | **0,72** | 0,45 | 0,42 |
| `nv` | 1.020 | 0,96 | 0,72 | 0,82 | 0,84 |
| `vasc` | 21 | 0,77 | 0,81 | 0,79 | 0,77 |

**Melanoma: más sensible, pero no demostrablemente mejor.** Con la regla de la clase
más probable, el modelo con CBAM detecta 115 de los 160 melanomas (recall 0,72) frente a
80 sin CBAM (0,50), y los melanomas confundidos con nevus bajan de 34 a 16. Pero también
responde "melanoma" con mucha más frecuencia: 346 veces frente a 217, y su precisión es
menor (0,33 frente a 0,37). Para separar los dos efectos se compararon medidas que no
dependen del punto de corte:

| Medida | Con CBAM | Sin CBAM |
| --- | --- | --- |
| Veces que predice melanoma | 346 | 217 |
| Recall con la regla de la clase más probable | 0,72 | 0,50 |
| Recall marcando las 217 imágenes más probables de melanoma | 0,53 | 0,48 |
| Recall marcando las 346 imágenes más probables de melanoma | 0,71 | 0,66 |
| AUC de melanoma frente al resto | 0,876 | 0,859 |

Con el mismo número de alertas, la ventaja de CBAM baja a 4–6 puntos de recall, y la
diferencia de AUC (+0,017, IC 95%: −0,001 a 0,036) no es significativa. La mayor parte del
aumento de recall viene de que CBAM **desplaza el punto de operación hacia el melanoma**:
detecta más, a costa de más falsas alarmas. Esa mayor sensibilidad podría ser útil en
triaje, pero también se obtendría bajando el umbral de melanoma del modelo sin CBAM.

Fuente: `reports/results/analisis_complementario.json` (`python -m src.eval.analisis_complementario`).

### 4.2 Segmentación y efecto del self-attention

| Modelo | Dice | IoU |
| --- | --- | --- |
| **U-Net sin self-attention** | **0,948** | **0,908** |
| U-Net + self-attention (modelo del aplicativo) | 0,945 | 0,905 |
| Diferencia (IC 95%) | −0,0024 (−0,0046 a −0,0005) | |

El self-attention empeora el Dice de forma estadísticamente significativa pero
prácticamente irrelevante: 0,24 puntos sobre 95. El modelo con atención da mayor Dice en
788 de las 1.502 imágenes, pero cuando falla lo hace por más (46 imágenes empeoran más de
0,05 y 29 mejoran más de 0,05). El aplicativo usa la U-Net con self-attention para mostrar
sus mapas de atención; la diferencia de 0,0024 no cambia la segmentación de forma visible.

**Por clase y tamaño de lesión** (U-Net con self-attention): el Dice medio es de 0,95 o
más en `nv` y `mel`, 0,93–0,94 en `bkl`, `df` y `vasc`, 0,90 en `akiec` y **0,84 en
`bcc`**. En lesiones que ocupan menos del 5% de la imagen baja a 0,92. Solo 27 de 1.502
imágenes tienen Dice menor a 0,7.

El Dice y el IoU se calculan por imagen a 256×256 px, la resolución de trabajo del modelo,
y luego se promedian. En 300 imágenes de test, calcularlo a la resolución original
(600×450) da prácticamente lo mismo: 0,9420 frente a 0,9417.

### 4.3 Localización de la atención del clasificador

Fracción del mapa de Grad-CAM que cae dentro de la **máscara anotada**, sobre las 1.502
imágenes de test:

| Modelo | Media | Mediana | Promedio por clase |
| --- | --- | --- | --- |
| Sin CBAM | 0,464 | 0,457 | 0,516 |
| Con CBAM | 0,443 | 0,437 | 0,480 |

CBAM no concentra más la atención en la lesión: lo hace solo en el 38,5% de las imágenes.
Sus cambios en el F1 por clase no se explican por mirar más la lesión.

El porcentaje que muestra el aplicativo es otra medida: usa la **máscara predicha** por el
segmentador, porque en uso real no hay anotación. No es comparable con esta tabla.

### 4.4 Optimización: cuantización INT8

Cuantización estática post-entrenamiento del clasificador final con ONNX Runtime,
calibrada con 200 imágenes de entrenamiento. Tolerancia definida de antemano: caída
máxima de 0,02 en macro-F1.

| Versión | Tamaño | Accuracy | Macro-F1 | Latencia CPU, 1 imagen |
| --- | --- | --- | --- | --- |
| FP32 (ONNX) | 85,3 MB | 0,720 | 0,650 | 40,1 ms |
| INT8 estático (ONNX) | 21,5 MB | 0,709 | 0,643 | 16,8 ms |
| Cambio | **−74,8%** | −0,011 | **−0,007**, dentro de la tolerancia | **2,4 veces más rápido** |

### 4.5 Tiempos de inferencia: CPU frente a GPU

| Dispositivo | Hardware | Software |
| --- | --- | --- |
| CPU | Intel Core i3-1305U (13.ª generación, 5 núcleos, 6 hilos), portátil conectado a la corriente, 5 hilos de PyTorch | Windows 11, PyTorch 2.14 |
| GPU | NVIDIA Tesla T4 (15,6 GB), Google Colab | Linux, PyTorch 2.11, CUDA 12.8 |

Precisión FP32, 10 pasadas de calentamiento y 50 mediciones. En CPU los modelos se
midieron intercalados en orden aleatorio, para que el calentamiento del procesador no
favoreciera a ninguno.

| Modelo | CPU, 1 imagen (mediana) | GPU, 1 imagen (mediana) | CPU, lote de 8 (por imagen) | GPU, lote de 8 (por imagen) |
| --- | --- | --- | --- | --- |
| Clasificador con CBAM | 82,6 ms | 5,7 ms | 69,4 ms | 2,2 ms |
| Clasificador sin CBAM | 79,7 ms | 5,6 ms | 68,0 ms | 2,2 ms |
| Segmentador con self-attention | 216,1 ms | 8,3 ms | 196,8 ms | 5,5 ms |
| Segmentador sin self-attention | 275,6 ms | 7,5 ms | 195,7 ms | 5,4 ms |
| Clasificador INT8 (ONNX Runtime, solo CPU) | 16,8 ms | — | 14,5 ms | — |

La GPU es entre 14 y 37 veces más rápida con una imagen, y unas 31–36 veces con lotes de
8. CBAM añade un 4% de tiempo en CPU. Con una imagen, la diferencia entre los dos
segmentadores en CPU está dentro del ruido de medición (sus medias son 245 y 256 ms); con
lotes de 8 tardan lo mismo. En el aplicativo, en el portátil de la tabla, el análisis
completo de una imagen tarda alrededor de 1 segundo.

Fuente: `reports/results/timings.csv`.

## 5. Alcances

El sistema se comporta como fue evaluado cuando:

- la imagen es **dermatoscópica**, tomada con dermatoscopio de contacto o de inmersión;
- contiene **una sola lesión**, visible y razonablemente centrada;
- la lesión pertenece a una de las **7 categorías** de HAM10000;
- la iluminación, el encuadre y la resolución son comparables a los de HAM10000, y la
  imagen no tiene ruido ni compresión fuerte.

En esas condiciones, la segmentación es sólida para lesiones de cualquier tamaño
(Dice medio entre 0,92 y 0,95 en todos los rangos de tamaño) y el clasificador detecta 72 de
cada 100 melanomas, con 2 falsas alarmas por cada acierto.

## 6. Limitaciones

**Del clasificador**

- **Siempre elige una de las 7 clases, sin opción de "no sé".** Con una imagen de piel
  sin ninguna lesión predice `nv` con 99,9% de confianza. El aplicativo lo advierte cuando
  el segmentador no encuentra lesión, pero el clasificador por sí solo no lo detecta.
- **Muy sensible al ruido.** En 300 imágenes de test, un ruido gaussiano del 2% del rango
  de la entrada cambia la clase predicha en el 28% de ellas, y uno del 5%, en el 45%, con
  una caída de accuracy de 0,71 a 0,57. Imágenes con ruido de sensor, compresión fuerte o
  mala iluminación pueden cambiar la predicción.
- **Muchas falsas alarmas de melanoma.** Precisión de 0,33: dos de cada tres alertas son
  falsas. Además, 16 de los 160 melanomas de test (10%) se clasificaron como nevus.
- **Clases con pocos ejemplos.** `df` y `vasc` tienen 19 y 21 imágenes de test: acertar o
  fallar dos imágenes puede cambiar su F1 en cerca de 0,1. Sus cifras deben leerse con cautela.
- **Probabilidades no calibradas.** La pérdida ponderada por clase desplaza las
  probabilidades hacia las clases minoritarias. Los porcentajes del aplicativo indican
  qué clase favorece el modelo, no la probabilidad real de enfermedad.

**Del segmentador**

- Es menos preciso en **carcinoma basocelular** (Dice 0,84) y **queratosis actínica**
  (0,90), cuyos bordes son menos definidos.
- Conserva solo la región conectada más grande: en una imagen con varias lesiones
  segmenta una.

**De los mecanismos de atención**

- Con comparaciones controladas, **ni CBAM ni el self-attention mejoran de forma
  relevante** las métricas globales de una ResNet-34 preentrenada en este problema. CBAM
  vuelve al clasificador más sensible al melanoma (más detecciones y más falsas alarmas),
  pero su mejora en la capacidad de distinguirlo (AUC +0,017) no es significativa.
- Los resultados provienen de **un solo entrenamiento por configuración** (semilla 42).
  Los intervalos de confianza reflejan la variación por la composición del test, no la
  variación entre entrenamientos.

**No evaluado**

- Fotografías clínicas o de teléfono celular: el aplicativo lo advierte al usar la cámara.
- Lesiones en mucosas, uñas o cuero cabelludo; piel pediátrica; lesiones tratadas o
  biopsiadas; imágenes desenfocadas o de baja resolución.
- Diagnósticos fuera de las 7 clases, que se asignan forzosamente a la más parecida.

## 7. Consideraciones éticas y de sesgo

- **Sesgo de fototipo cutáneo.** Es la limitación más importante. HAM10000 proviene de
  poblaciones de Austria y Australia, con predominio de fototipos claros (Fitzpatrick
  I–III); los fototipos IV–VI están subrepresentados, y la apariencia dermatoscópica de
  las lesiones cambia con la pigmentación de la piel. **El desempeño reportado no es
  transferible a la población colombiana sin una nueva validación**, y podría ser peor en
  piel más oscura, justo donde el melanoma suele diagnosticarse más tarde. El dataset no
  registra el fototipo, así que este sesgo se declara pero no se puede medir.
- **Etiquetas de distinta solidez.** Las clases malignas están confirmadas por
  histopatología en el 100% de los casos; parte de las benignas se basa en seguimiento
  clínico o consenso de expertos, que son referencias menos firmes.
- **Riesgo de falsa tranquilidad.** El daño principal de un sistema así es que alguien
  posponga una consulta por un resultado "benigno". Por eso el aplicativo muestra en todo
  momento que no es un dispositivo diagnóstico.
- **Errores de distinto costo.** No detectar un melanoma es mucho más grave que una falsa
  alarma, aunque la accuracy cuente ambos errores igual. Por eso se reportan el recall de
  melanoma y la matriz de confusión, además de las métricas agregadas.
- **Datos y privacidad.** Licencia CC BY-NC 4.0: uso académico con atribución, sin uso
  comercial. Las imágenes de HAM10000 están anonimizadas en la fuente. El aplicativo
  procesa las imágenes en memoria y no las guarda. Cuando se publica con Cloudflare Tunnel,
  la imagen viaja cifrada (HTTPS), pero pasa por los servidores de Cloudflare antes de
  llegar al computador que ejecuta los modelos.

## 8. Reproducibilidad

| Elemento | Valor |
| --- | --- |
| Semilla | 42 (`configs/paths.yaml`), fijada también después de construir cada modelo |
| Hardware de entrenamiento | NVIDIA Tesla T4 en Google Colab |
| Duración del entrenamiento | Clasificador: 42 min con CBAM y 45 min sin él. Segmentador: 61 min con self-attention y 60 min sin él |
| Entorno de entrenamiento | Python 3.13, PyTorch 2.11 con CUDA 12.8 |
| Entorno local | Python 3.12, dependencias fijadas en `requirements.lock.txt` |
| Notebooks ejecutados | `notebooks/ejecutados/`, con todas las salidas |
| Resultados | `reports/results/` |

El entrenamiento es determinista: al volver a entrenar el clasificador sin CBAM con la
misma semilla, las primeras 12 épocas reprodujeron exactamente los valores de la corrida
anterior, hasta el cuarto decimal.

## Referencias

- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Jacob, B., Kligys, S., Chen, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*.
- Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). Model cards for model reporting. *FAT\* 2019*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS*.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *ECCV*.

# DermaScope: clasificación, segmentación y atención en imágenes dermatoscópicas

**Proyecto 2 · Visión Computacional con Deep Learning**
Maestría en Inteligencia Artificial y Ciencia de Datos — Universidad Autónoma de Occidente

**Integrantes:** Maria Stella Fuentes Diaz · Sergio Luis Castaño Rodríguez · Alejandro Galvez Cardenas · Joan Sebastian Mena Ortega

**Repositorio:** https://github.com/mariastella1807/dermascope · **Fecha:** 18 de septiembre de 2026

## Resumen

DermaScope clasifica imágenes dermatoscópicas en 7 categorías diagnósticas, segmenta y aísla
la lesión, y explica cada predicción con Grad-CAM. Se entrenaron sobre HAM10000 una
ResNet-34 con CBAM (macro-F1 de 0,650 en test) y una U-Net con encoder ResNet-34 y un bloque
de self-attention (Dice de 0,945). Cada mecanismo de atención se evaluó con una comparación
controlada contra la misma arquitectura sin él y con intervalos de confianza por bootstrap:
ninguno mejora de forma relevante las métricas globales, pero CBAM eleva el recall de
melanoma de 0,50 a 0,72. La cuantización INT8 reduce el clasificador un 74,8% y lo acelera
2,4 veces en CPU con una caída de macro-F1 de 0,007. La GPU es entre 14 y 37 veces más rápida
que la CPU. El sistema se despliega como aplicativo Streamlit publicado con Cloudflare Tunnel.

## 1. Problemática

La dermatoscopia es una técnica de imagen no invasiva que, con aumento óptico e iluminación
controlada, hace visibles estructuras pigmentadas de la epidermis y la dermis superficial.
Su interpretación exige entrenamiento y varía entre observadores, lo que motiva sistemas de
apoyo para el triaje de lesiones pigmentadas: identificar cuáles requieren valoración
prioritaria por un especialista.

El problema es difícil por tres razones:

- **Clases visualmente parecidas.** Lesiones benignas y malignas comparten rasgos; la
  confusión más relevante es entre nevus y melanoma.
- **Desbalance.** Como en la práctica clínica, predominan las lesiones benignas: los nevus
  son el 67% de las imágenes y el dermatofibroma el 1,1%.
- **Condiciones de captura.** Contraste bajo entre lesión y piel, bordes difusos y
  artefactos como vello, burbujas de inmersión o el viñeteado del dermatoscopio.

El sistema resuelve tres tareas sobre la misma imagen: **clasificación** en 7 clases,
**segmentación** de la lesión con su aislamiento del fondo, e **interpretabilidad**
mediante atención. Es una herramienta académica, no un dispositivo diagnóstico.

## 2. Datos

Se usa **HAM10000** (Tschandl et al., 2018): 10.015 imágenes dermatoscópicas de 600×450 px,
de 7.470 lesiones, provenientes de Viena (Austria) y Queensland (Australia), con licencia
CC BY-NC 4.0. Todas las imágenes tienen máscara binaria de la lesión, lo que permite
resolver clasificación y segmentación sobre las mismas imágenes.

La partición 70/15/15 (7.014 / 1.499 / 1.502 imágenes) se hace por `lesion_id`,
estratificada por diagnóstico y con semilla 42. Es necesaria porque 1.956 lesiones tienen
varias fotografías: repartir por imagen pondría la misma lesión en entrenamiento y en test,
y el modelo la reconocería en lugar de generalizar. Se verificó que ninguna lesión queda en
dos particiones. La distribución completa por clase y partición está en la model card.

## 3. Decisiones de arquitectura

| Decisión | Alternativa descartada | Razón |
| --- | --- | --- |
| **ResNet-34** preentrenada en ImageNet como clasificador | EfficientNet-B0 | Sus etapas `layer1` a `layer4` son secuenciales, así que CBAM se inserta dentro del backbone y no después del pooling global. Con 21 M de parámetros se entrena en unos 40 minutos en una GPU T4 |
| **CBAM** al final de `layer3` y `layer4` | En todas las etapas | Las etapas tempranas codifican bordes y texturas de bajo nivel y tienen mapas grandes; reponderar ahí aporta poco y cuesta más. Añade solo 41 mil parámetros (+0,19%) |
| **U-Net con encoder ResNet-34** para segmentar | Un Transformer de segmentación completo | Permite comparar con y sin atención cambiando un solo bloque; un Transformer completo cambiaría toda la arquitectura. Comparte familia de encoder con el clasificador |
| **Self-attention en el cuello de botella** (8×8 = 64 tokens) | En etapas de mayor resolución | El costo de la atención crece con el cuadrado de los tokens: en `layer1` la matriz tendría 16,8 millones de entradas por imagen, 1.024 veces más que en `layer4` |
| **Segmentación a 256 px** | 512 px | La lesión ocupa una mediana del 21,9% de la imagen, y en el percentil 10 aún mide unos 63×63 px a 256 px. Entrenar a 512 px costaría unas 4 veces más |
| **Macro-F1** para seleccionar el clasificador | Accuracy | Un clasificador que siempre responde "nevus" logra 0,679 de accuracy y 0,116 de macro-F1 |
| **Entropía cruzada ponderada** por la frecuencia inversa de cada clase | Sobremuestreo | No duplica imágenes de las clases raras, lo que favorecería memorizarlas |
| **Pérdida BCE + Dice** | BCE sola | Ante una predicción "todo fondo", la BCE ya es baja (0,29 en el ejemplo del notebook 02) mientras que el término Dice queda cerca de su peor valor (0,99) |
| **Tasas de aprendizaje diferenciadas** (×10 para capas nuevas) | Una sola tasa | Una tasa alta destruiría lo aprendido en ImageNet; una baja haría avanzar muy lento a las capas que parten de cero |
| **25 épocas completas, sin parada temprana** | Parada temprana | Con parada temprana, una inestabilidad pasajera detuvo una corrida antes que su par de comparación (sección 7). El checkpoint conserva la mejor época, que es lo que controla el sobreajuste |
| **Cuantización INT8 estática** | Cuantización dinámica | La estática fija de antemano las escalas de las activaciones con datos de calibración; la dinámica las calcula en cada inferencia, con un costo adicional en cada ejecución |

## 4. Resultados principales

Todas las cifras son del conjunto de test. Cada comparación con y sin atención comparte
semilla, partición, aumentaciones, hiperparámetros, número de épocas y pesos iniciales de
las capas comunes. Los intervalos de confianza del 95% se obtienen por bootstrap pareado
sobre las predicciones de cada imagen (10.000 remuestreos).

### 4.1 Clasificación y CBAM

| Modelo | Accuracy | Accuracy balanceada | Macro-F1 | Recall de melanoma |
| --- | --- | --- | --- | --- |
| ResNet-34 | 0,725 | 0,726 | 0,638 | 0,50 |
| ResNet-34 + CBAM | 0,720 | 0,746 | 0,650 | 0,72 |
| Diferencia (IC 95%) | −0,005 (−0,025 a 0,014) | +0,020 (−0,009 a 0,051) | +0,012 (−0,019 a 0,043) | +0,22 (McNemar p = 1,8·10⁻⁷) |

CBAM no mejora de forma significativa las métricas globales: por clase, sube el F1 de
`akiec`, `bkl`, `mel` y `vasc`, y baja el de `bcc`, `df` y `nv`. Su efecto claro está en el
**melanoma**, la clase de mayor consecuencia clínica: detecta 115 de 160 frente a 80, y los
melanomas confundidos con nevus bajan de 34 a 16. El precio es la precisión de melanoma, de
0,33.

### 4.2 Segmentación y self-attention

| Modelo | Dice | IoU |
| --- | --- | --- |
| U-Net | 0,948 | 0,908 |
| U-Net + self-attention | 0,945 | 0,905 |
| Diferencia (IC 95%) | −0,0024 (−0,0046 a −0,0005) | |

El self-attention empeora el Dice de forma significativa pero despreciable en la práctica:
0,24 puntos sobre 95. Una explicación plausible es que la atención aporta poco que la red no
tuviera: al llegar a `layer4`, el campo receptivo de la ResNet-34 ya cubre toda la imagen,
y las máscaras de referencia, semiautomáticas, dejan poco margen por encima de 0,95. La
segmentación es más débil en carcinoma basocelular (Dice 0,84) y en lesiones que ocupan
menos del 5% de la imagen (0,92).

### 4.3 Localización de la atención

Fracción del mapa de Grad-CAM que cae dentro de la máscara anotada, en las 1.502 imágenes
de test: 0,464 sin CBAM y 0,443 con CBAM. CBAM no hace que el clasificador mire más la
lesión, lo que ocurre solo en el 38,5% de las imágenes; su efecto sobre el melanoma se
explica por cómo pondera la lesión y la piel que la rodea, no por concentrarse en la lesión.

## 5. Proceso de optimización

Se optimizó el clasificador final (ResNet-34 + CBAM) con **cuantización INT8 estática
post-entrenamiento**, que representa pesos y activaciones con enteros de 8 bits en lugar de
números de 32 bits (Jacob et al., 2018):

1. **Exportación a ONNX** (opset 17).
2. **Calibración** con 200 imágenes reales de entrenamiento, procesadas con la misma
   transformación de evaluación que verá el modelo en uso. La calibración observa el rango
   de valores de cada activación para fijar sus escalas; con imágenes no representativas,
   como ruido aleatorio, las escalas quedarían mal ajustadas.
3. **Cuantización** con ONNX Runtime: pesos en `int8` y activaciones en `uint8`.
4. **Evaluación** de las versiones FP32 e INT8 sobre el mismo conjunto de test, contra una
   tolerancia fijada antes de medir: caída máxima de 0,02 en macro-F1.

| Versión | Tamaño en disco | Accuracy | Macro-F1 | CPU, 1 imagen | CPU, lote de 8 (por imagen) |
| --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 85,5 MB | 0,720 | 0,650 | 82,6 ms | 69,4 ms |
| ONNX FP32 | 85,3 MB | 0,720 | 0,650 | 40,1 ms | 37,6 ms |
| **ONNX INT8** | **21,5 MB** | 0,709 | **0,643** | **16,8 ms** | **14,5 ms** |

El modelo INT8 pesa un 74,8% menos y es 2,4 veces más rápido que el mismo modelo en ONNX
FP32, y 4,9 veces más rápido que en PyTorch, con una caída de macro-F1 de 0,007, dentro de
la tolerancia. El F1 por clase cambia entre −0,05 (`akiec`) y +0,03 (`df`). Que las
métricas de ONNX FP32 coincidan exactamente con las de PyTorch confirma que la exportación
es correcta.

El aplicativo sigue usando el modelo PyTorch FP32, porque Grad-CAM necesita calcular
gradientes y el modelo INT8 de ONNX Runtime no los ofrece. La ganancia del INT8, unos 65 ms
por imagen, sería imperceptible en un análisis que tarda alrededor de un segundo.

## 6. Tiempos de ejecución: CPU frente a GPU

| Dispositivo | Hardware | Software |
| --- | --- | --- |
| CPU | Intel Core i3-1305U (13.ª generación, 5 núcleos, 6 hilos), portátil conectado a la corriente, 5 hilos de PyTorch | Windows 11, PyTorch 2.14 |
| GPU | NVIDIA Tesla T4 (15,6 GB), Google Colab | Linux, PyTorch 2.11, CUDA 12.8 |

Mediana de 50 mediciones en FP32, en milisegundos:

| Modelo | CPU, 1 imagen | GPU, 1 imagen | Aceleración | CPU, lote de 8 (por imagen) | GPU, lote de 8 (por imagen) | Aceleración |
| --- | --- | --- | --- | --- | --- | --- |
| Clasificador + CBAM | 82,6 | 5,7 | 14,5× | 69,4 | 2,2 | 31× |
| Clasificador sin CBAM | 79,7 | 5,6 | 14,2× | 68,0 | 2,2 | 31× |
| Segmentador + self-attention | 216,1 | 8,3 | 25,9× | 196,8 | 5,5 | 36× |
| Segmentador sin self-attention | 275,6 | 7,5 | 36,8× | 195,7 | 5,4 | 36× |

- La ventaja de la GPU crece con el lote, porque procesa muchas imágenes en paralelo; con
  una sola imagen no alcanza a ocupar su capacidad.
- CBAM añade un 4% de tiempo en CPU. Entre los dos segmentadores, la diferencia con una
  imagen está dentro del ruido de medición (sus medias en CPU son 245 y 256 ms), y con
  lotes de 8 tardan lo mismo: el bloque de atención opera sobre solo 64 tokens.
- En el aplicativo, en este portátil, el análisis completo de una imagen (clasificación,
  Grad-CAM, segmentación y aislamiento) tarda alrededor de 1 segundo.

**Método de medición.** Diez pasadas de calentamiento, para excluir la carga inicial de
kernels; `torch.cuda.synchronize()` antes de detener el cronómetro, porque las llamadas a
la GPU son asíncronas; y mediana y percentil 95 además de la media. En CPU hubo que resolver
dos problemas de medición que producían números incoherentes:

1. Medir los modelos uno tras otro, o rotando solo el punto de partida, dejaba a un modelo
   siempre después del más pesado, con el procesador ya caliente: el segmentador sin
   atención aparecía más lento que el que sí la tiene. Se pasó a **intercalar los modelos
   en orden aleatorio en cada ronda**.
2. Los hilos de ONNX Runtime quedan en espera activa después de cada inferencia y ocupan
   núcleos: el modelo medido justo después parecía hasta 10 veces más lento. PyTorch y
   ONNX Runtime se miden ahora **en grupos separados**, y ONNX Runtime sin espera activa.

## 7. Problemas encontrados y cómo se resolvieron

| Problema | Cómo se detectó | Solución |
| --- | --- | --- |
| **Comparación de CBAM sesgada por el número de épocas.** La corrida sin CBAM sufrió una inestabilidad en la época 8 (pérdida de validación de 4,60) y la parada temprana la detuvo en la época 12, mientras la corrida con CBAM completó 25. La ventaja de CBAM era +0,097 en macro-F1 | Al comparar las curvas: hasta la época 12, el modelo sin CBAM iba ganando | Parada temprana desactivada y corrida sin CBAM repetida con 25 épocas. La ventaja bajó a +0,012, no significativa. Las primeras 12 épocas se reprodujeron exactamente, lo que confirma que el entrenamiento es determinista |
| **Desbordamiento numérico en el self-attention.** Con precisión mixta, `QKᵀ` se calculaba en float16, cuyo máximo es 65.504. En la época 8 la pérdida pasó a NaN y el Dice a 0 | La pérdida y el Dice en el registro de entrenamiento | `QKᵀ`, el softmax y la combinación con `V` se calculan en float32; el resto de la red sigue en float16. Además, el entrenamiento se detiene si la pérdida deja de ser finita, y un estado guardado con NaN se descarta en vez de continuarse |
| **Mapas de SmoothGrad-CAM vacíos en el aplicativo.** SmoothGrad promedia Grad-CAM sobre copias de la imagen con ruido | El aplicativo mostraba 0% de atención en un melanoma que ocupaba el 80% de la imagen | Se midió que un ruido del 2% cambia la clase predicha en el 28% de las imágenes de test. El aplicativo usa Grad-CAM directo, que además baja el tiempo por imagen de 2 a 1 segundo. La sensibilidad al ruido quedó documentada como limitación en la model card |

## 8. Despliegue

El aplicativo Streamlit (`app/streamlit_app.py`) acepta una imagen subida, una foto de la
cámara o una de siete imágenes de ejemplo. Los ejemplos son la primera imagen de cada clase
en el split de test, elegidas sin mirar si el modelo acierta, y se muestran junto a su
diagnóstico de referencia. Para cada imagen presenta la clase predicha con las
probabilidades de las 7 clases, la máscara de segmentación, la lesión aislada sobre fondo
transparente y recortada a su caja, el mapa de Grad-CAM con la fracción de atención dentro
de la lesión, y la model card.

La interfaz advierte sus límites: al usar la cámara, que el modelo solo conoce imágenes de
dermatoscopio; y si el segmentador no encuentra lesión, que la clasificación no es
confiable, porque el clasificador siempre elige una de las 7 clases (con una imagen de piel
sin lesión responde "nevus" con 100% de confianza).

Para la sustentación, `scripts/run_tunnel.ps1` levanta el aplicativo y abre un túnel rápido
de Cloudflare, que entrega una dirección pública `https://….trycloudflare.com` sin cuenta
ni configuración de red. La dirección cambia en cada ejecución, así que se genera el mismo
día. El despliegue se probó abriendo la dirección pública desde un teléfono.

## 9. Conclusiones y trabajo futuro

- Una ResNet-34 y una U-Net preentrenadas resuelven bien el problema: macro-F1 de 0,65
  sobre 7 clases muy desbalanceadas y Dice de 0,95.
- Con comparaciones controladas y pruebas estadísticas, **los mecanismos de atención no
  mejoraron de forma relevante las métricas globales**. El aporte de CBAM se concentra en
  el recall de melanoma, un resultado clínicamente relevante que el promedio esconde.
- La cuantización INT8 es una optimización efectiva: cuatro veces menos espacio y casi
  cinco veces menos tiempo en CPU, con pérdida mínima de desempeño.
- Varios hallazgos vinieron de revisar resultados que no cuadraban, no de las métricas
  finales: la comparación sesgada por épocas, el desbordamiento en float16 y los errores de
  medición de tiempos.

**Trabajo futuro**, en orden de prioridad:

1. **Validar en fototipos IV–VI**, subrepresentados en HAM10000; el desempeño reportado no
   es transferible a la población colombiana sin esa validación.
2. **Repetir las comparaciones con varias semillas**, para separar el efecto de la atención
   de la variación entre entrenamientos.
3. **Robustez al ruido**, con aumentaciones de ruido y compresión en el entrenamiento.
4. **Detección de imágenes fuera de distribución** y **calibración de probabilidades**,
   para que el sistema pueda responder "no sé".
5. **Ajustar el umbral de melanoma** según el costo clínico de cada tipo de error.

## Referencias

- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Jacob, B., Kligys, S., Chen, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*.
- Micikevicius, P., Narang, S., Alben, J., et al. (2018). Mixed precision training. *ICLR*.
- Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS*.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *ECCV*.

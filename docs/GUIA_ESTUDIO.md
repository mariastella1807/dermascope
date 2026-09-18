# Guía de estudio para la sustentación

> **Para qué sirve.** El profesor escoge **al azar** quién presenta, y si esa persona no
> puede responder, el grupo saca 0. Esta guía reúne lo que **cualquiera de los cuatro**
> debe poder explicar: el proyecto de principio a fin, cada técnica en palabras simples,
> los resultados con su interpretación y las preguntas más probables, con dónde está la
> evidencia de cada respuesta.
>
> Como el resto del repositorio, es una **propuesta en revisión**. Si algo no se entiende
> o no están de acuerdo, hay que corregirlo antes de la sustentación.

**Contenido**

1. [La historia del proyecto en 2 minutos](#1-la-historia-del-proyecto-en-2-minutos)
2. [Los datos](#2-los-datos)
3. [Tarea 1: clasificación](#3-tarea-1-clasificación)
4. [Tarea 2: segmentación y aislamiento](#4-tarea-2-segmentación-y-aislamiento)
5. [Tarea 3: atención](#5-tarea-3-atención)
6. [Cómo se comparó con y sin atención](#6-cómo-se-comparó-con-y-sin-atención)
7. [Optimización: cuantización INT8](#7-optimización-cuantización-int8)
8. [Tiempos CPU frente a GPU](#8-tiempos-cpu-frente-a-gpu)
9. [La app y el túnel](#9-la-app-y-el-túnel)
10. [Problemas que aparecieron y cómo se resolvieron](#10-problemas-que-aparecieron-y-cómo-se-resolvieron)
11. [Limitaciones que hay que saber reconocer](#11-limitaciones-que-hay-que-saber-reconocer)
12. [Preguntas probables del profesor](#12-preguntas-probables-del-profesor)
13. [Números clave para memorizar](#13-números-clave-para-memorizar)
14. [Consejos para responder](#14-consejos-para-responder)

---

## 1. La historia del proyecto en 2 minutos

Si solo recuerdas una sección, que sea esta.

1. **El problema.** La dermatoscopia permite ver estructuras de la piel que no se ven a
   simple vista, pero interpretarla requiere experiencia. Un sistema de apoyo puede
   ayudar a decidir qué lesiones revisar primero (triaje). No es un diagnóstico.
2. **Los datos.** HAM10000: 10.015 imágenes dermatoscópicas de 7 tipos de lesión, cada una
   con la máscara que marca la lesión. El 67% son nevus (lunares comunes): los datos están
   muy desbalanceados.
3. **Las tres tareas sobre la misma imagen:**
   - **Clasificar** la lesión en 7 clases, con una ResNet-34 preentrenada y CBAM.
   - **Segmentar** la lesión con una U-Net con self-attention, y **aislarla** del fondo.
   - **Atención:** comparar cada modelo con y sin su bloque de atención, y mostrar con
     Grad-CAM en qué se fija el clasificador.
4. **Los resultados.**
   - Clasificación: macro-F1 de 0,650.
   - Segmentación: Dice de 0,945.
   - Con comparaciones justas y pruebas estadísticas, **la atención no mejora de forma
     relevante las métricas globales**.
5. **Optimización.** Con cuantización INT8, el clasificador pesa 4 veces menos y es casi 5
   veces más rápido en CPU, perdiendo apenas 0,007 de macro-F1.
6. **Despliegue.** Una app en Streamlit, publicada con Cloudflare Tunnel, que muestra las
   tres tareas para cualquier imagen.
7. **Lo que diferencia al proyecto.** Se detectaron y corrigieron errores propios con
   evidencia: una comparación injusta, un desbordamiento numérico y una conclusión
   exagerada sobre el melanoma.

## 2. Los datos

| Dato | Valor |
| --- | --- |
| Imágenes | 10.015, de 600×450 píxeles |
| Lesiones distintas | 7.470; 1.956 tienen más de una foto |
| Clases | 7: `nv` 66,9%, `mel` 11,1%, `bkl` 11,0%, `bcc` 5,1%, `akiec` 3,3%, `vasc` 1,4%, `df` 1,1% |
| Máscaras | Todas las imágenes, semiautomáticas y revisadas manualmente |
| Fuente | Harvard Dataverse, DOI 10.7910/DVN/DBW86T, licencia CC BY-NC 4.0 |
| Partición | 70/15/15 → 7.014 entrenamiento, 1.499 validación, 1.502 test |

**Idea clave: la partición se hace por lesión, no por imagen.** Como una misma lesión puede
tener varias fotos, si las fotos se repartieran al azar, fotos de la misma lesión quedarían
en entrenamiento y en test. El modelo "reconocería" la lesión en lugar de aprender, y el
resultado de test saldría inflado. Por eso se agrupa por `lesion_id` y se verifica que
ninguna lesión quede en dos particiones.
*Dónde mostrarlo:* notebook 01, §1.1.

**Idea clave: el desbalance.** Con 67% de nevus, un modelo que siempre responde "nevus"
obtiene 0,679 de accuracy sin aprender nada. Por eso:

- se mide con **macro-F1**, que promedia las 7 clases por igual;
- la pérdida **pondera** cada clase por el inverso de su frecuencia: un error en
  dermatofibroma (peso 2,78) cuesta 60 veces más que uno en nevus (peso 0,046).

*Dónde mostrarlo:* notebook 01, §1.4 y §3.1.

## 3. Tarea 1: clasificación

**Modelo.** ResNet-34 preentrenada en ImageNet, con **fine-tuning**: se parte de una red
que ya sabe reconocer bordes, texturas y formas, y se ajusta a las lesiones de piel.

- **ResNet** usa conexiones residuales ("atajos" que suman la entrada de un bloque a su
  salida), que permiten entrenar redes profundas.
- **Tasas de aprendizaje diferenciadas:** el backbone preentrenado aprende con 3·10⁻⁴ y las
  capas nuevas (la capa final y CBAM) con una tasa 10 veces mayor. Una tasa alta en el
  backbone destruiría lo que aprendió en ImageNet.

**Entrenamiento.**

| Componente | Qué es |
| --- | --- |
| AdamW | Optimizador, con weight decay separado del gradiente |
| ReduceLROnPlateau | Si la pérdida de validación no mejora en 2 épocas, reduce la tasa de aprendizaje a la mitad |
| Épocas | 25 completas, sin parada temprana (ver §10) |
| Precisión mixta | Cálculos en float16 donde se puede: más rápido en GPU |
| Imagen | 224×224 px, lote de 32 |
| Selección del modelo | La época con mejor macro-F1 **en validación**. El test no se usa para elegir nada |

**Métricas.**

- **Accuracy:** porcentaje de aciertos.
- **F1** de una clase: combina precisión (de lo que marqué como clase X, cuánto lo era) y
  recall (de lo que era clase X, cuánto encontré): F1 = 2PR/(P+R).
- **Macro-F1:** el promedio del F1 de las 7 clases, cada una con el mismo peso.

**Resultados en test (modelo final, con CBAM):** accuracy 0,720, accuracy balanceada 0,746,
macro-F1 0,650. El mejor F1 es de `nv` (0,82) y el peor de `mel` (0,45).
*Dónde mostrarlo:* notebook 01, §4.

## 4. Tarea 2: segmentación y aislamiento

**Modelo: U-Net.** Tiene forma de "U":

- el **encoder** (una ResNet-34 preentrenada) reduce la imagen a la mitad cinco veces, de
  256×256 a 8×8, para entender *qué* hay;
- el **decoder** la vuelve a agrandar hasta 256×256 para decir *dónde* está;
- las **skip connections** pasan detalle del encoder al decoder en cada escala, y son las
  que permiten un borde preciso.

**Métricas.** Dice = 2|P∩G| / (|P|+|G|) e IoU = |P∩G| / |P∪G|: miden cuánto se solapan la
máscara predicha P y la real G, entre 0 y 1. Se calculan por imagen y se promedian.

**Pérdida: BCE + Dice.** La BCE (entropía cruzada binaria) evalúa píxel por píxel. Si la
lesión es pequeña, un modelo que diga "todo es fondo" ya tiene BCE baja (0,29 en el ejemplo
del notebook), mientras que el término Dice queda cerca de su peor valor (0,99). Sumar los
dos obliga al modelo a encontrar la lesión.
*Dónde mostrarlo:* notebook 02, §2.2.

**Resultado:** Dice de 0,945 e IoU de 0,905 en test. Es más débil en carcinoma basocelular
(0,84) y en lesiones que ocupan menos del 5% de la imagen (0,92).

**Aislamiento de la lesión** (lo exige el enunciado). A partir de la máscara:

1. **Cierre morfológico:** rellena huecos pequeños, por ejemplo cuando un pelo cruza la
   lesión. Se hace con max pooling (dilatar) seguido de max pooling negado (erosionar).
2. **Región más grande:** se queda con la mancha conectada más grande y descarta el ruido.
3. **Recorte:** se usa la máscara como transparencia y se recorta a la caja de la lesión,
   sobre la imagen **original**.

*Dónde mostrarlo:* notebook 02, §8, y en la app, pestaña "Lesión aislada".

## 5. Tarea 3: atención

El enunciado pide **dos** componentes: un self-attention integrado en un modelo, y CBAM
y/o Grad-CAM mostrado en la app. El proyecto tiene los tres.

### CBAM (dentro del clasificador)

Repondera el mapa de características en dos pasos:

1. **Atención de canal: ¿qué filtros importan?** Resume cada canal con su promedio y su
   máximo, los pasa por una pequeña red y obtiene un peso entre 0 y 1 por canal.
2. **Atención espacial: ¿en qué zona de la imagen?** Resume los canales en cada posición y,
   con una convolución 7×7, obtiene un peso por píxel.

Se ubica al final de `layer3` y `layer4`, las etapas profundas, y solo añade 41 mil
parámetros (+0,19%).
*Dónde mostrarlo:* notebook 01, §2.1 y §2.2, donde CBAM se escribe paso a paso y se
comprueba que coincide con el del proyecto.

### Self-attention (dentro de la U-Net)

Es la operación del Transformer y del Vision Transformer:

**Atención(Q, K, V) = softmax(Q·Kᵀ / √d) · V**

- El mapa de 8×8 que sale del encoder se trata como **64 "tokens"** de 512 números.
- Cada token genera una **consulta (Q)**, una **clave (K)** y un **valor (V)**.
- Q·Kᵀ mide cuánto se parece cada token a los demás; el softmax lo convierte en pesos que
  suman 1; y cada token se reemplaza por la mezcla de los V de todos, según esos pesos.
- **Por qué sirve:** una convolución 3×3 solo ve a sus vecinos; la atención conecta cada
  posición con **todas** las demás en un solo paso.
- **Por qué en el cuello de botella:** el costo crece con el cuadrado de los tokens. En
  8×8 la matriz tiene 4.096 entradas; en 64×64 tendría 16,8 millones, 1.024 veces más.

*Dónde mostrarlo:* notebook 02, §4.2 (implementación paso a paso) y §7 (mapas de atención).

### Grad-CAM (interpretabilidad, en la app)

Muestra en qué zonas de la imagen se apoyó el clasificador para su predicción:

1. Toma las activaciones de la última etapa convolucional (`layer4`) y el gradiente de la
   clase predicha respecto a ellas.
2. El promedio del gradiente de cada canal dice cuánto aporta ese canal a la clase.
3. El mapa es la suma de los canales ponderada por esos pesos, pasada por ReLU para
   quedarse solo con lo que apoya a la clase.

*Dónde mostrarlo:* notebook 01, §5, y en la app, pestaña "Atención (Grad-CAM)".

## 6. Cómo se comparó con y sin atención

**Comparación justa:** la versión con atención y la versión sin ella comparten todo lo
demás: semilla, partición, aumentaciones, hiperparámetros, 25 épocas, pesos iniciales de
las capas comunes y el orden de los lotes. Así, la diferencia solo puede venir del bloque.

**Prueba estadística: bootstrap pareado.** El test tiene solo 19 imágenes de `df` y 21 de
`vasc`, así que acertar o fallar unas pocas mueve bastante el macro-F1. Para saber si una
diferencia es real o azar:

1. se sortean 1.502 imágenes del test **con reemplazo** (algunas se repiten, otras no salen);
2. sobre **las mismas** imágenes se calcula la métrica de los dos modelos y su diferencia;
3. se repite 10.000 veces.

El 95% central de esas diferencias es el **intervalo de confianza**: si incluye el 0, no se
puede afirmar que un modelo sea mejor. *Lo que no mide:* cuánto cambiaría el resultado con
otra semilla de entrenamiento, porque se entrenó una sola vez cada modelo.

**Resultados:**

| Comparación | Diferencia | Intervalo del 95% | Conclusión |
| --- | --- | --- | --- |
| CBAM, macro-F1 | +0,012 | −0,019 a +0,043 | No significativa |
| Self-attention, Dice | −0,0024 | −0,0046 a −0,0005 | Significativa, pero despreciable: 0,24 puntos sobre 95 |

**El caso del melanoma (hay que entenderlo bien).** Con CBAM, el recall de melanoma sube de
0,50 a 0,72. Parece un gran aporte, pero el modelo con CBAM también **responde "melanoma"
mucho más seguido** (346 veces frente a 217) y su precisión baja (0,33 frente a 0,37).

Si se compara a los dos modelos con **el mismo número de alertas**, la ventaja baja a 4–6
puntos, y la diferencia de AUC, una medida que no depende del umbral, no es significativa.
Conclusión: CBAM hace al modelo **más sensible** al melanoma, pero no hay evidencia de que
lo **distinga mejor**. Lo mismo se lograría bajando el umbral del modelo sin CBAM.
*Dónde mostrarlo:* `reports/results/analisis_complementario.json` y model card §4.1.

**Grad-CAM con y sin CBAM:** la fracción del mapa dentro de la lesión es 0,464 sin CBAM y
0,443 con CBAM: CBAM **no** hace que el modelo mire más la lesión.

**Por qué la atención aporta poco aquí (hipótesis, no demostrada):** la ResNet-34
preentrenada ya "ve" toda la imagen al llegar a `layer4`, las máscaras semiautomáticas dejan
poco margen por encima de un Dice de 0,95, y hay una sola lesión por imagen, casi siempre
grande y centrada.

## 7. Optimización: cuantización INT8

**Qué es.** Guardar pesos y cálculos con números enteros de 8 bits en lugar de números
decimales de 32 bits. El modelo ocupa unas 4 veces menos y los cálculos con enteros son más
rápidos en CPU, a cambio de perder algo de precisión.

**Cómo se hizo:**

1. Exportar el clasificador final a **ONNX**, un formato estándar de modelos.
2. **Calibrar** con 200 imágenes reales de entrenamiento: el modelo "mira" datos reales
   para saber en qué rango se mueven los valores de cada capa y fijar la escala de los
   enteros. Con datos no representativos, como ruido, las escalas quedarían mal.
3. Cuantizar con ONNX Runtime (estática: las escalas se fijan de antemano).
4. Comparar FP32 e INT8 sobre el mismo test, con una tolerancia fijada **antes** de medir:
   perder como máximo 0,02 de macro-F1.

| Versión | Tamaño | Macro-F1 | Tiempo en CPU, 1 imagen |
| --- | --- | --- | --- |
| ONNX FP32 | 85,3 MB | 0,650 | 40,1 ms |
| ONNX INT8 | 21,5 MB | 0,643 | 16,8 ms |

**Resultado:** −74,8% de tamaño, 2,4 veces más rápido y −0,007 de macro-F1, dentro de la
tolerancia.
*Dónde mostrarlo:* notebook 01, §6.

**Por qué la app no usa el modelo INT8:** Grad-CAM necesita calcular gradientes y el modelo
de ONNX Runtime no los ofrece. Además, ahorraría unos 65 ms en un análisis de 1 segundo.

## 8. Tiempos CPU frente a GPU

| Dispositivo | Hardware |
| --- | --- |
| CPU | Intel Core i3-1305U (portátil) |
| GPU | NVIDIA Tesla T4 (Google Colab) |

| Modelo | CPU, 1 imagen | GPU, 1 imagen | La GPU es… |
| --- | --- | --- | --- |
| Clasificador con CBAM | 82,6 ms | 5,7 ms | 14,5 veces más rápida |
| Segmentador con self-attention | 216,1 ms | 8,3 ms | 26 veces más rápida |

Con lotes de 8 imágenes la GPU es de 31 a 36 veces más rápida, porque procesa muchas en
paralelo.

**Cómo se midió bien:**

- **Calentamiento:** 10 pasadas antes de medir, porque las primeras incluyen tiempos de
  carga.
- **Sincronizar la GPU** antes de parar el cronómetro, porque la GPU trabaja de forma
  asíncrona.
- **50 mediciones**, reportando la mediana y el percentil 95.
- En CPU, los modelos se midieron **intercalados en orden aleatorio**, para que el
  calentamiento del procesador no perjudicara a ninguno.

*Dónde mostrarlo:* notebook 01, §7 (GPU), y `reports/results/timings.csv`.

**Punto débil conocido:** con una imagen en CPU, el segmentador con atención sale un poco
más rápido que el que no la tiene, lo cual no tiene sentido. La diferencia está dentro del
ruido de medición, y con lotes de 8 tardan lo mismo. Si lo preguntan, reconocerlo así.

## 9. La app y el túnel

**Streamlit** convierte un script de Python en una página web. La app:

- recibe una imagen subida, una foto de la cámara o una de 7 imágenes de ejemplo;
- muestra la clase predicha con las probabilidades de las 7 clases, la máscara, la lesión
  aislada, el mapa de Grad-CAM y la model card;
- tarda alrededor de 1 segundo por imagen en un portátil, sin GPU.

**Cloudflare Tunnel** le da a la app, que corre en un portátil, una dirección pública
`https://….trycloudflare.com` para abrirla desde cualquier celular. La dirección cambia cada
vez y solo funciona mientras el portátil esté prendido.

**Avisos que tiene la app:**

- Al usar la cámara, advierte que una foto de celular no es una imagen dermatoscópica.
- Si no detecta lesión, advierte que la clasificación no es confiable.

Cómo usarla: `docs/GUIA_REVISION.md`, sección 6.

## 10. Problemas que aparecieron y cómo se resolvieron

Esta sección muestra dominio del tema: son errores que el equipo detectó y corrigió con
evidencia.

| Problema | Cómo se detectó | Solución |
| --- | --- | --- |
| **Comparación de CBAM injusta.** El modelo sin CBAM se inestabilizó en la época 8, y la parada temprana lo cortó en la 12, mientras el de CBAM entrenó 25 épocas. CBAM parecía ganar por +0,097 | Hasta la época 12, el modelo sin CBAM iba ganando | Se quitó la parada temprana y se reentrenó con 25 épocas: la ventaja bajó a +0,012. Las primeras 12 épocas salieron **idénticas** a la corrida anterior, lo que demuestra que el entrenamiento es reproducible |
| **La pérdida se volvió NaN en la época 8 del segmentador con self-attention** | El registro de entrenamiento: pérdida `nan` y Dice 0 | En float16 el número más grande representable es 65.504, y Q·Kᵀ lo superó. Ahora esa operación se calcula en float32, y el entrenamiento se detiene solo si vuelve a aparecer un NaN |
| **SmoothGrad-CAM daba mapas vacíos en la app** | 0% de atención en un melanoma que ocupaba el 80% de la imagen | SmoothGrad promedia mapas de copias con ruido, y el clasificador cambia de clase con muy poco ruido. Se usa Grad-CAM directo |
| **La conclusión del melanoma estaba exagerada** | Una auditoría buscó explicaciones alternativas | Se comparó con igual número de alertas y con AUC (§6) |
| **Tiempos incoherentes en CPU** | Un modelo aparecía 10 veces más lento que otro casi igual | Los hilos de ONNX Runtime ocupaban la CPU; se midieron por separado y en orden aleatorio |

## 11. Limitaciones que hay que saber reconocer

- **No sabe decir "no sé":** con una imagen sin lesión responde "nevus" con 99,9% de
  confianza.
- **Sensible al ruido:** un ruido del 2% cambia la predicción en el 28% de las imágenes.
- **Muchas falsas alarmas de melanoma:** precisión de 0,33.
- **Pocas imágenes de `df` y `vasc`** en test (19 y 21): sus cifras son poco estables.
- **Sesgo de piel:** los datos son de Austria y Australia, sobre todo de piel clara. No se
  puede suponer que funcione igual en población colombiana.
- **Una sola semilla:** no se midió cuánto varían los resultados entre entrenamientos.
- **Solo imágenes de dermatoscopio:** no fotos de celular.

*Dónde:* model card §6 y §7.

## 12. Preguntas probables del profesor

**Sobre datos y métricas**

1. **¿Por qué no usaron accuracy como métrica principal?**
   Porque con 67% de nevus, responder siempre "nevus" da 0,679 de accuracy y solo 0,116 de
   macro-F1. El macro-F1 obliga al modelo a aprender las 7 clases.
   *Evidencia:* notebook 01, §1.4.
2. **¿Cómo evitaron la fuga de datos?**
   Partiendo por lesión (`lesion_id`) y no por imagen, y verificando que ninguna lesión
   quede en dos particiones.
   *Evidencia:* notebook 01, §1.1.
3. **¿Cómo manejaron el desbalance?**
   Con pérdida ponderada por el inverso de la frecuencia de cada clase, y midiendo con
   macro-F1.
   *Evidencia:* notebook 01, §3.1.

**Sobre los modelos**

4. **¿Por qué ResNet-34?**
   Sus etapas permiten insertar CBAM dentro de la red, se entrena en unos 40 minutos en
   una T4, y se usa la misma familia en las dos tareas.
   *Evidencia:* informe, §3.
5. **¿Por qué el self-attention va en el cuello de botella?**
   Porque su costo crece con el cuadrado de los tokens: con 64 tokens es barato; en una
   etapa de mayor resolución sería 1.024 veces más caro.
   *Evidencia:* notebook 02, §4.4.
6. **¿Cómo saben que el self-attention hace algo y no es decorativo?**
   Participa en el forward y sus pesos se entrenaron: su proyección de salida empezó en
   cero y terminó con valores distintos de cero. Además, sus mapas se visualizan.
   *Evidencia:* notebook 02, §4.2 y §7.
7. **¿Qué es la U-Net y para qué las skip connections?**
   Ver §4 de esta guía.

**Sobre la atención**

8. **Si la atención no mejora, ¿para qué la pusieron?**
   El enunciado pide medir qué aporta frente a una versión sin ella, y eso se hizo con
   comparaciones justas y pruebas estadísticas. El resultado honesto es que en este
   problema aporta poco. Hipótesis: la red preentrenada ya ve toda la imagen, y las
   máscaras dejan poco margen.
9. **¿Es cierto que CBAM detecta mejor el melanoma?**
   Detecta más (recall de 0,72 frente a 0,50), pero porque predice melanoma más seguido.
   Con el mismo número de alertas la ventaja es de 4 a 6 puntos, y su AUC no es
   significativamente mayor. Es más sensible, no demostrablemente mejor.
   *Evidencia:* `analisis_complementario.json`.
10. **¿Qué es un bootstrap pareado y qué no mide?**
    Ver §6. No mide la variación entre entrenamientos con distinta semilla.
11. **¿Cómo se lee un mapa de Grad-CAM?**
    Las zonas rojas son las que más apoyaron la predicción de la clase; las azules, las que
    menos.

**Sobre optimización y tiempos**

12. **¿Por qué cuantización y no poda?**
    La cuantización reduce el tamaño unas 4 veces sin reentrenar, y la pérdida fue mínima
    (0,007). La poda suele necesitar reentrenamiento para recuperar desempeño.
13. **¿Para qué sirve la calibración?**
    Para fijar las escalas de los enteros según los valores reales que toma cada capa.
    Se hace con imágenes reales.
14. **¿Por qué la GPU es más rápida y por qué más con lotes grandes?**
    Tiene miles de núcleos que operan en paralelo; con una sola imagen no alcanza a
    ocuparlos todos.
15. **¿Cómo se aseguraron de medir bien los tiempos?**
    Calentamiento, sincronización de la GPU, 50 mediciones con mediana y percentil 95, y en
    CPU orden aleatorio.

**Sobre el proceso y la app**

16. **¿Qué pasó en la época 8?** Ver §10: en el clasificador sin CBAM, una inestabilidad;
    en el segmentador, el desbordamiento de float16.
17. **¿Qué pasa si subo una foto de mi brazo?**
    La app advierte que el modelo solo conoce imágenes de dermatoscopio. Si no encuentra
    lesión, advierte que la clasificación no es confiable.
18. **¿Por qué la app tiene un modelo de segmentación con atención si el que no la tiene es
    un poco mejor?**
    Para poder mostrar los mapas de self-attention. La diferencia (0,0024 de Dice) no se ve
    en la segmentación.
19. **¿Esto se puede usar con pacientes?**
    No. No está validado clínicamente y tiene sesgo de piel (§11).
20. **¿Qué harían con más tiempo?**
    Validar en pieles más oscuras, repetir con varias semillas, entrenar con ruido para
    hacerlo más robusto, y enseñarle a decir "no sé" cuando la imagen no se parece a las de
    entrenamiento.

## 13. Números clave para memorizar

| Qué | Valor |
| --- | --- |
| Imágenes / lesiones | 10.015 / 7.470 |
| Nevus en el dataset | 67% |
| Partición | 70/15/15 por lesión |
| Macro-F1 del clasificador (con CBAM / sin CBAM) | 0,650 / 0,638 |
| Accuracy del clasificador | 0,720 |
| Dice del segmentador (con atención / sin atención) | 0,945 / 0,948 |
| Recall de melanoma (con CBAM / sin CBAM) | 0,72 / 0,50 |
| Parámetros que añade CBAM / self-attention | +0,19% / +4,4% |
| INT8: tamaño y velocidad | −74,8% y 2,4 veces más rápido |
| INT8: pérdida de macro-F1 | 0,007 (tolerancia 0,02) |
| GPU frente a CPU, 1 imagen | 14 a 37 veces más rápida |
| Tiempo por imagen en la app, en portátil | ≈ 1 segundo |
| Épocas | 25 |

## 14. Consejos para responder

- **Si no sabes algo, dilo y muestra dónde está la evidencia** en lugar de inventar. Por
  ejemplo: "No lo recuerdo exacto, pero está en el notebook 01, sección 4; lo muestro".
- **Reconoce las limitaciones antes de que te las señalen.** Un resultado negativo bien
  explicado vale más que uno positivo exagerado.
- **Distingue lo medido de lo supuesto.** Por ejemplo: "medimos que no mejora; *creemos*
  que es porque la red ya ve toda la imagen, pero no lo demostramos".
- **Ten abiertos antes de empezar:** los dos notebooks ejecutados, la model card, la app y
  esta guía.

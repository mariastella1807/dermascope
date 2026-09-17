# Informe — DermaScope

Proyecto 2 · Visión Computacional con Deep Learning — Maestría en Inteligencia Artificial y Ciencia de Datos, Universidad Autónoma de Occidente

**Integrantes:** Maria Stella Fuentes Diaz · Sergio Luis Castaño Rodríguez · Alejandro Galvez Cardenas · Joan Sebastian Mena Ortega

> Documento breve exigido por §5: problemática abordada, decisiones de arquitectura y
> proceso de optimización, incluyendo las tablas de tiempos CPU vs GPU.
> Estructura para llenar a medida que avanzan los experimentos.

## 1. Problemática

- Contexto clínico: por qué importa el triaje de lesiones pigmentadas.
- Por qué es un problema de visión: qué señal visual discrimina las clases (regla ABCD).
- Qué resuelve el sistema y qué explícitamente no resuelve.

## 2. Dataset

- Fuente y licencia (ver `scripts/download_data.md`).
- Tabla de distribución de clases con los conteos **reales** de `build_splits.py`.
- Justificación del agrupamiento por `lesion_id`: incluir el número de lesiones únicas
  y explicar la fuga que se evita.
- Cobertura de máscaras: cuántas de las 10.015 imágenes tienen anotación de segmentación.

## 3. Decisiones de arquitectura

Para cada decisión, la alternativa que se descartó y por qué. Es la parte que más se
pregunta en la sustentación.

| Decisión | Alternativa descartada | Razón |
| --- | --- | --- |
| ResNet-34 como backbone | EfficientNet-B0 | `layer1..4` son `nn.Sequential`, así que CBAM se integra dentro del backbone y no detrás del pooling; además cuantiza a INT8 sin sorpresas |
| U-Net + bloque de self-attention | SegFormer (transformer completo) | La ablación con y sin el bloque cambia un solo factor, mientras que comparar SegFormer contra U-Net cambia toda la arquitectura; además mantiene el mismo encoder que el clasificador |
| Segmentación a 256px | 512px | Las lesiones ocupan buena parte de la imagen y el entrenamiento es ~4 veces más rápido |
| `ReduceLROnPlateau` | Coseno con warmup | Reacciona a la curva de validación real en lugar de seguir un cronograma fijo |
| Macro-F1 como métrica de selección | Accuracy | Con `nv` al 67%, la accuracy premia al clasificador constante |
| Pérdida ponderada por clase | Oversampling | No duplica imágenes ni altera la distribución de las aumentaciones |
| BCE + Dice | BCE sola | La BCE sola sesga hacia el fondo, que domina el área en lesiones pequeñas |
| INT8 estático | Cuantización dinámica | La dinámica solo afecta capas lineales y ResNet es casi toda convolución |

## 4. Mecanismos de atención (§4.3)

Cada mecanismo con su comparación propia.

### 4.1 CBAM en el clasificador

- Dónde se insertó y por qué solo en las etapas profundas.
- Tabla de ablación: `reports/results/ablacion_cbam.csv`.
- Discusión: ¿la mejora en macro-F1 es consistente o se concentra en pocas clases?
- Costo: parámetros añadidos y latencia añadida (de `timings.csv`).

### 4.2 Self-attention en el segmentador

- Dónde va el bloque (cuello de botella, 8x8 = 64 tokens) y por qué ahí: la convolución
  solo ve vecinos; la auto-atención conecta cada posición con todas las demás.
- Tabla U-Net con y sin self-attention: `reports/results/comparacion_segmentacion.csv`.
- Mapa de atención: `SelfAttention2d.last_attention` guarda la matriz 64x64 para
  visualizar a qué regiones mira cada posición.

### 4.3 Grad-CAM como interpretabilidad

- Figuras comparativas: mismo caso con y sin CBAM.
- Tabla cuantitativa de masa de atención dentro de la lesión:
  `reports/results/localizacion_atencion.csv`.
- Casos de fallo: imágenes donde la atención se va al vello o al viñeteado.

## 5. Proceso de optimización (§4.4)

- Técnica aplicada y por qué esa.
- Papel del conjunto de calibración: qué se calibra y qué pasa si se usa ruido en vez
  de imágenes reales.
- Tabla antes/después con **tamaño en disco** y métricas: `reports/results/optimization.json`.
- Si la caída de macro-F1 excedió la tolerancia, documentar la ruta de poda.

## 6. Tiempos de ejecución (§4.5)

Fuente: `reports/results/timings.csv`.

| Modelo | Precisión | Dispositivo | Hardware | Batch | ms/imagen | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| | | CPU | | 1 | | |
| | | GPU | | 1 | | |

Notas metodológicas a incluir: warmup, `torch.cuda.synchronize()`, número de
repeticiones y por qué se reporta p95 además de la media.

## 7. Despliegue (§4.6)

- Flujo del aplicativo y capturas.
- Procedimiento del Cloudflare Tunnel el día de la sustentación.

## 8. Conclusiones y trabajo futuro

- Qué funcionó, qué no, y qué se haría distinto con más tiempo.
- La limitación de fototipo cutáneo como línea de trabajo prioritaria.

# Guía para revisar DermaScope

> **Estado: propuesta en revisión.** Este repositorio es una propuesta de Proyecto 2 para
> discutir en equipo; no es la versión final. Todo se puede cuestionar, corregir o
> cambiar. Esta guía explica dónde está cada cosa y cómo revisarla, desde solo leer hasta
> ejecutar la app en tu computador.

**Contenido**

1. [Qué es, en cinco minutos](#1-qué-es-en-cinco-minutos)
2. [Mapa del repositorio](#2-mapa-del-repositorio)
3. [Dónde está cada requisito del enunciado](#3-dónde-está-cada-requisito-del-enunciado)
4. [Ver los resultados sin ejecutar nada](#4-ver-los-resultados-sin-ejecutar-nada)
5. [Ejecutar los notebooks en Colab](#5-ejecutar-los-notebooks-en-colab)
6. [Usar la app en tu computador](#6-usar-la-app-en-tu-computador)
7. [Si quieres modificar la app](#7-si-quieres-modificar-la-app)
8. [Qué revisar](#8-qué-revisar)
9. [Cómo proponer cambios](#9-cómo-proponer-cambios)

---

## 1. Qué es, en cinco minutos

DermaScope analiza **imágenes dermatoscópicas**: fotos de lunares y lesiones de piel
tomadas con un dermatoscopio, que es un lente de aumento con luz controlada. Usa el dataset
público **HAM10000**: 10.015 imágenes de 7 tipos de lesión, cada una con la máscara que
marca dónde está la lesión.

Sobre la misma imagen resuelve las tres tareas del enunciado:

| Tarea | Qué hace | Modelo |
| --- | --- | --- |
| Clasificación | Dice qué tipo de lesión es, entre 7 (nevus, melanoma, etc.) | ResNet-34 preentrenada + **CBAM** |
| Segmentación | Marca el borde de la lesión y la separa de la piel | U-Net + **self-attention** |
| Atención | Compara cada modelo con y sin su bloque de atención, y muestra con **Grad-CAM** en qué se fija el clasificador | — |

Además tiene cuantización INT8 del clasificador, medición de tiempos CPU vs GPU, una app en
Streamlit que se publica con Cloudflare Tunnel, la model card y un documento breve.

**Resultados en el conjunto de test**

- Clasificación: macro-F1 0,650 y accuracy 0,720.
- Segmentación: Dice 0,945.
- Con comparaciones justas, **ni CBAM ni self-attention mejoran de forma relevante** las
  métricas globales. Es un resultado que hay que saber explicar, y los documentos lo
  analizan.

## 2. Mapa del repositorio

Todo está en https://github.com/mariastella1807/dermascope. Para abrir una carpeta o un
archivo, haz clic en su nombre en la página principal.

| Dónde | Qué contiene | Para qué mirarlo |
| --- | --- | --- |
| [`notebooks/`](../notebooks) | `01_entrenar_clasificador_colab.ipynb` y `02_entrenar_segmentador_colab.ipynb`, **sin ejecutar** | Son los que se abren en Colab para correr todo |
| [`notebooks/ejecutados/`](../notebooks/ejecutados) | Los mismos dos notebooks **ya ejecutados**, con todas las salidas y gráficas | Ver los resultados sin correr nada. Es el mejor punto de partida |
| [`MODEL_CARD.md`](../MODEL_CARD.md) | Problema, dataset, arquitecturas, métricas, alcances, limitaciones y ética | Entregable obligatorio |
| [`reports/informe.md`](../reports/informe.md) | Documento breve: problemática, decisiones de arquitectura, optimización y tablas CPU vs GPU | Entregable obligatorio |
| [`reports/results/`](../reports/results) | Los archivos de resultados que generó el código (`.json` y `.csv`) | De aquí salen todos los números de los documentos |
| [`app/`](../app) | La app de Streamlit: `streamlit_app.py` (interfaz), `inference.py` (modelos) y `assets/` (7 imágenes de ejemplo) | Para probarla o pulirla |
| [`src/models/`](../src/models) | `attention.py` (CBAM y self-attention), `classifier.py`, `segmenter.py` | Las arquitecturas |
| [`src/train/`](../src/train) | Los scripts de entrenamiento | Cómo se entrenó |
| [`src/eval/`](../src/eval) | Métricas, comparaciones con y sin atención, tiempos (`benchmark.py`) y análisis adicionales | Cómo se evaluó |
| [`src/explain/`](../src/explain) | `gradcam.py` (Grad-CAM) y `isolate.py` (aislar la lesión) | La interpretabilidad y el recorte |
| [`src/optimize/`](../src/optimize) | `quantize.py`, la cuantización INT8 | La optimización |
| [`configs/`](../configs) | Hiperparámetros de cada tarea (`.yaml`) | Épocas, tasas de aprendizaje, tamaños |
| [`scripts/`](../scripts) | Instalación, descarga de modelos, túnel y verificaciones | Para usar la app |
| [Releases → `modelos-v1`](https://github.com/mariastella1807/dermascope/releases/tag/modelos-v1) | Los 6 modelos entrenados (no caben en git) | Los descargan solos los notebooks y el script de la sección 6 |

## 3. Dónde está cada requisito del enunciado

| Requisito | Dónde verlo |
| --- | --- |
| Dataset: fuente y conformación | Notebook 01 §0.7 y §1 · `MODEL_CARD.md` §2 · `scripts/download_data.md` |
| Clasificación con accuracy y F1 (§4.1) | Notebook 01 §3 y §4 · `reports/results/classification_cbam.json` |
| Segmentación con Dice e IoU (§4.2) | Notebook 02 §2, §5 y §6 · `reports/results/segmentation_attn.json` |
| Aislar la lesión y mostrarla en la app (§4.2) | Notebook 02 §8 · app, pestaña "Lesión aislada" · `src/explain/isolate.py` |
| Self-attention integrado (§4.3) | Notebook 02 §4.2 y §4.3 · `src/models/attention.py` (`SelfAttention2d`) |
| CBAM integrado y Grad-CAM en la app (§4.3) | Notebook 01 §2 y §5 · app, pestaña "Atención (Grad-CAM)" |
| Comparación con y sin cada bloque (§4.3) | Notebook 01 §3.4 y §4 · notebook 02 §5.3 y §6 · `reports/results/ablacion_cbam*.csv` y `comparacion_segmentacion*.csv` |
| Optimización antes/después (§4.4) | Notebook 01 §6 · `reports/results/optimization.json` · informe §5 |
| Tiempos CPU vs GPU con hardware (§4.5) | Notebook 01 §7 (GPU) · `reports/results/timings.csv` · informe §6 |
| App en Streamlit con imagen o cámara (§4.6) | `app/streamlit_app.py` · sección 6 de esta guía |
| Cloudflare Tunnel (§4.6) | `scripts/run_tunnel.ps1` · sección 6.5 de esta guía |
| Model card (§4.7) | `MODEL_CARD.md`, también en la pestaña "Model card" de la app |
| Documento breve (§5) | `reports/informe.md` |

## 4. Ver los resultados sin ejecutar nada

Los notebooks ejecutados se leen mejor en **nbviewer**, que muestra siempre las gráficas.
GitHub a veces no puede mostrar notebooks tan grandes.

- **Clasificación**, con CBAM, Grad-CAM, cuantización y tiempos en GPU:
  https://nbviewer.org/github/mariastella1807/dermascope/blob/main/notebooks/ejecutados/01_entrenar_clasificador_colab.ipynb
- **Segmentación**, con self-attention y aislamiento de la lesión:
  https://nbviewer.org/github/mariastella1807/dermascope/blob/main/notebooks/ejecutados/02_entrenar_segmentador_colab.ipynb

Cada notebook empieza con una guía "Qué buscar | Dónde está", explica en un texto antes de
cada celda qué se hace y por qué, y termina con un checklist de verificación.

## 5. Ejecutar los notebooks en Colab

Solo necesitas una cuenta de Google. No hay que instalar nada ni entrenar: los notebooks
descargan los modelos ya entrenados.

1. Abre el notebook:
   - Clasificador: https://colab.research.google.com/github/mariastella1807/dermascope/blob/main/notebooks/01_entrenar_clasificador_colab.ipynb
   - Segmentador: https://colab.research.google.com/github/mariastella1807/dermascope/blob/main/notebooks/02_entrenar_segmentador_colab.ipynb
2. Menú **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU → Guardar**. Es
   necesario: el notebook 01 mide tiempos en GPU.
3. Menú **Entorno de ejecución → Ejecutar todas**.
4. Cuando aparezca la ventana de Google Drive, acepta. El notebook crea una carpeta
   `dermascope` en **tu** Drive y guarda ahí los resultados; no toca el Drive de nadie más.

**Qué va a pasar:**

| Paso | Qué ves |
| --- | --- |
| §0.7 y §0.8 | Descarga y descomprime el dataset: 2,6 GB, unos 2 minutos |
| §0.11 | `descargando classifier_cbam_best.pt del release ... verificado` |
| Celdas de entrenamiento | `Ya entrenado (25 epocas): se usa el resultado guardado en Drive` |
| Final | `RESULTADO FINAL: TODAS LAS VERIFICACIONES PASAN` |

Tarda unos 15 minutos por notebook. Vas a ver **los mismos números** que en los
ejecutados, porque se usan exactamente los mismos modelos.

**Si algo falla:**

- *"No hay GPU"*: repite el paso 2.
- *Colab se desconectó*: vuelve a usar *Ejecutar todas*; lo que ya estaba hecho no se repite.
- *Quieres entrenar desde cero*: borra de tu Drive el archivo `.json` del entrenamiento
  (`dermascope/reports/results/`). Tarda unos 45 minutos por modelo.

## 6. Usar la app en tu computador

La app **no necesita el dataset**: solo el código y dos modelos (unos 190 MB). Corre en la
CPU, así que no hace falta una tarjeta gráfica.

### 6.1 Lo que necesitas

- **Python 3.11 o 3.12**, que son las versiones que busca el script de instalación.
  Descárgalo de https://www.python.org/downloads/; en Windows, marca la casilla
  *"Add python.exe to PATH"* al instalar. Puedes tenerlo junto a otras versiones.
- **Git**: https://git-scm.com/downloads
- Unos 3 GB libres y conexión a internet.

### 6.2 Windows (PowerShell)

Abre **PowerShell** en la carpeta donde quieras el proyecto y ejecuta uno por uno:

```powershell
git clone https://github.com/mariastella1807/dermascope.git
cd dermascope
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_env.ps1
.\.venv\Scripts\python.exe scripts\descargar_modelos.py
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Qué hace cada línea:

1. Descarga el código.
2. Entra a la carpeta.
3. Permite ejecutar los scripts del proyecto **solo en esta ventana**; al cerrarla, vuelve a
   la configuración normal.
4. Crea un entorno de Python aislado (`.venv`) e instala las librerías. Tarda entre 5 y 10
   minutos la primera vez.
5. Descarga los dos modelos que usa la app y verifica que llegaron completos.
6. Abre la app.

Cuando la terminal muestre `Local URL: http://localhost:8501`, abre esa dirección en el
navegador. Para cerrar la app, vuelve a la terminal y presiona **Ctrl+C**.

**La próxima vez** solo necesitas la última línea, desde la carpeta `dermascope`.

### 6.3 Mac o Linux (Terminal)

```bash
git clone https://github.com/mariastella1807/dermascope.git
cd dermascope
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/descargar_modelos.py
python -m streamlit run app/streamlit_app.py
```

Si tienes Python 3.11, usa `python3.11` en la tercera línea. La próxima vez: entra a la
carpeta, activa el entorno con `source .venv/bin/activate` y ejecuta la última línea.

> Las instrucciones de Windows se probaron de principio a fin en Windows 11, clonando el
> repositorio desde cero. En Mac y Linux son los pasos estándar de Python, pero no se han
> probado en este proyecto.

### 6.4 Comprobar que todo quedó bien

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

En Mac o Linux: `python scripts/smoke_test.py`. Debe terminar con **`11 pasaron, 0 fallaron`**.

### 6.5 Publicarla para que otros la abran desde el celular

1. Instala Cloudflare Tunnel: en Windows, `winget install --id Cloudflare.cloudflared`; en
   Mac, `brew install cloudflared`.
2. Con la app corriendo, abre **otra** terminal y ejecuta:

   ```
   cloudflared tunnel --url http://localhost:8501
   ```

   En Windows también sirve `.\scripts\run_tunnel.ps1`, que abre la app y el túnel juntos.
3. En la salida aparece una dirección `https://….trycloudflare.com`. Cualquiera que la abra
   ve **tu** app, que corre en **tu** computador.

**Ten en cuenta:** la dirección cambia cada vez, deja de funcionar al cerrar la terminal o
suspender el computador, y cualquiera que la tenga puede entrar mientras está abierta.

### 6.6 Problemas frecuentes

| Problema | Solución |
| --- | --- |
| `py` o `python` no se reconoce | Python no está en el PATH: reinstálalo marcando *"Add python.exe to PATH"* |
| `setup_env.ps1` dice que no encuentra Python 3.11 ni 3.12 | Instala Python 3.12 (sección 6.1) |
| `no se puede cargar el archivo … porque la ejecución de scripts está deshabilitada` | Falta la línea 3 de la sección 6.2 (`Set-ExecutionPolicy …`) |
| La app dice `Falta el checkpoint …` | Falta el paso 5: `scripts\descargar_modelos.py` |
| `descargar_modelos.py` dice que un archivo llegó incompleto | Se cortó la descarga: vuelve a ejecutarlo |
| El puerto 8501 está ocupado | Ya hay una app abierta; ciérrala o usa `--server.port 8502` |
| `git clone` falla con `Filename too long` | La carpeta donde clonas tiene una ruta muy larga (pasa a veces dentro de OneDrive). Clona en una carpeta corta, por ejemplo `C:\proyectos` |

## 7. Si quieres modificar la app

| Archivo | Qué cambiar ahí |
| --- | --- |
| `app/streamlit_app.py` | Todo lo visual: textos, pestañas, avisos, orden de los resultados |
| `app/inference.py` | Qué modelos se cargan y en qué orden se ejecuta el análisis |
| `app/assets/` | Las imágenes de ejemplo (ver su `README.md`) |

Con la app abierta, al guardar un cambio en `streamlit_app.py` aparece un botón
**Rerun** arriba a la derecha de la página: al pulsarlo se ven los cambios, sin reiniciar.
Después de modificar, corre `scripts/smoke_test.py` para comprobar que nada se rompió.

**Ideas que valdría la pena revisar:**

- Los textos de la interfaz: ¿se entienden para alguien que no sabe del proyecto?
- Cómo se ve en el celular: la barra lateral se esconde y hay que abrirla con la flecha **›**.
- Con fotos grandes de celular (12 MP) el análisis tarda unos 5 segundos; ¿reducirlas antes?
- ¿Qué pasa si varias personas la usan a la vez? Nunca se ha probado.

## 8. Qué revisar

**Contenido**

- [ ] ¿La propuesta cumple todos los requisitos del enunciado? (tabla de la sección 3)
- [ ] ¿Las conclusiones de la model card y el informe son claras, y estamos de acuerdo con
      cómo se presentan? En especial, que la atención no mejore las métricas globales.
- [ ] ¿Cada número de los documentos coincide con `reports/results/`?

**Preparación para la sustentación.** El profesor escoge **al azar** quién presenta, y si
esa persona no puede responder, el grupo saca 0. La **[guía de estudio](GUIA_ESTUDIO.md)**
explica cada punto y trae 20 preguntas probables con su respuesta. Cada integrante debería
poder explicar:

- [ ] Por qué la partición se hace por lesión y no por imagen.
- [ ] Qué hace CBAM y qué hace self-attention, con sus fórmulas.
- [ ] Qué es Grad-CAM y cómo se lee el mapa.
- [ ] Qué es el Dice y por qué la pérdida combina BCE y Dice.
- [ ] Qué es la cuantización INT8 y qué se ganó.
- [ ] Por qué la GPU es más rápida y cuánto.
- [ ] Qué mide el bootstrap de las comparaciones y qué no mide.
- [ ] Los problemas que aparecieron y cómo se resolvieron (informe §7).

**Lo que todavía no existe**

- [ ] Diapositivas del pitch de 7 minutos.
- [ ] Guion de los 7 minutos de código y demo.
- [ ] Video de respaldo de la demo, por si falla internet.
- [ ] Ensayo del túnel desde la universidad con varios celulares.

**Pendientes externos**

- [ ] Confirmar la fecha de entrega: el enunciado dice semana 8 y la presentación del curso,
      24/26 de septiembre.
- [ ] Después de la clase de optimización (semana 7), revisar si el profesor espera otra
      técnica distinta de la cuantización.

## 9. Cómo proponer cambios

- **Comentarios y dudas:** por el grupo, citando el archivo y la sección.
- **Cambios en el código o en los documentos:** para subirlos al repositorio hay que tener
  permiso de colaborador. Pásale tu usuario de GitHub a Maria Stella para que te agregue.
  Después: haz tus cambios, revisa que `scripts/smoke_test.py` pase y súbelos con
  `git add`, `git commit` y `git push`. Si otra persona subió cambios antes, primero
  ejecuta `git pull`.
- **Nunca** uses `git push --force`: puede borrar el trabajo de los demás.

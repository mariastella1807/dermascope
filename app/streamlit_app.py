"""Aplicativo Streamlit (§4.6).

El enunciado es explicito en que no se evalua el diseno visual, sino que el flujo
funcione de principio a fin sin errores. Asi que la interfaz es deliberadamente plana:
una fuente de imagen y las salidas obligatorias en pestanas.

Salidas que el enunciado exige mostrar aqui:
  - clasificacion con probabilidades (§4.1)
  - mascara de segmentacion (§4.2)
  - lesion aislada del fondo (§4.2, requisito explicito)
  - mapa de Grad-CAM (§4.3)
  - model card accesible desde la interfaz (§4.7)

Ejecucion:
    streamlit run app/streamlit_app.py
    .\\scripts\\run_tunnel.ps1        # expone la URL publica
"""

from __future__ import annotations

import sys
from pathlib import Path

# El app se ejecuta como script, no como modulo, asi que hay que poner la raiz del repo
# en el path para que `import src...` funcione.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app.inference import DermaPipeline  # noqa: E402

# Imagenes de test de HAM10000, una por clase: <clase>_<image_id>.jpg (ver assets/README.md).
EXAMPLES_DIR = Path(__file__).parent / "assets"

st.set_page_config(page_title="DermaScope", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos ...")
def get_pipeline() -> DermaPipeline:
    """Una sola carga de modelos por proceso.

    Sin `cache_resource`, Streamlit reejecuta el script completo en cada interaccion y
    recargaria el clasificador y la U-Net desde disco cada vez, lo que hace la demo inusable.
    """
    return DermaPipeline()


def read_image(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


pipeline = get_pipeline()

st.title("DermaScope — Análisis dermatoscópico asistido")
st.caption(
    "Herramienta académica de apoyo. **No es un dispositivo diagnóstico** y no "
    "sustituye la valoración de un dermatólogo."
)

with st.sidebar:
    st.header("Entrada")
    source = st.radio("Fuente de la imagen", ["Imagen de ejemplo", "Subir archivo", "Cámara"])
    reference_label = None

    if source == "Imagen de ejemplo":
        examples = sorted(EXAMPLES_DIR.glob("*.jpg"))
        chosen = st.selectbox(
            "Imagen de test de HAM10000",
            examples,
            format_func=lambda p: (
                f"{p.stem.split('_', 1)[0]} — "
                f"{pipeline.class_names_es.get(p.stem.split('_', 1)[0], '')} ({p.stem.split('_', 1)[1]})"
            ),
        )
        file = chosen
        reference_label = chosen.stem.split("_", 1)[0] if chosen else None
        st.caption(
            "Primera imagen de cada clase en el split de test: los modelos no la vieron al "
            "entrenar. Licencia CC BY-NC 4.0 (Tschandl et al., 2018)."
        )
    elif source == "Subir archivo":
        file = st.file_uploader("Imagen dermatoscópica", type=["jpg", "jpeg", "png"])
    else:
        st.warning(
            "El modelo se entrenó solo con imágenes de **dermatoscopio**, que usan aumento e "
            "iluminación controlada. Una foto de celular es un tipo de imagen distinto: los "
            "resultados sobre ella no son confiables."
        )
        file = st.camera_input("Captura")

    st.divider()
    st.caption(f"Dispositivo de inferencia: `{pipeline.device}`")

if file is None:
    st.info("Elige una imagen de ejemplo o carga una imagen dermatoscópica para iniciar el análisis.")
    st.stop()

image = read_image(file)

with st.spinner("Ejecutando clasificación, segmentación y Grad-CAM ..."):
    result = pipeline.run(image)

# --- Resumen -----------------------------------------------------------------
label_es = pipeline.class_names_es.get(result.predicted_class, result.predicted_class)
confidence = result.probabilities[result.predicted_class]
detected = result.segmentation.detected

columns = st.columns(4 if reference_label else 3)
columns[0].metric("Clase predicha", result.predicted_class.upper(), label_es, delta_color="off")
columns[1].metric("Confianza", f"{confidence:.1%}")
columns[2].metric(
    "Área de la lesión",
    f"{result.segmentation.area_ratio:.1%}" if detected else "—",
    "de la imagen" if detected else None,
    delta_color="off",
)
if reference_label:
    acierto = "coincide" if reference_label == result.predicted_class else "no coincide"
    columns[3].metric(
        "Diagnóstico de referencia",
        reference_label.upper(),
        f"{pipeline.class_names_es.get(reference_label, '')}: {acierto}",
        delta_color="normal" if acierto == "coincide" else "inverse",
    )

if not detected:
    st.warning(
        "La segmentación no encontró una región de lesión con el área mínima esperada. "
        "Puede que la imagen no contenga una lesión visible o que no sea dermatoscópica; "
        "por eso no se muestran la lesión aislada ni la medida de atención. **La "
        "clasificación tampoco es confiable:** el modelo siempre elige una de las 7 clases, "
        "aunque la imagen no tenga ninguna lesión."
    )

tab_cls, tab_seg, tab_iso, tab_attn, tab_card = st.tabs(
    ["Clasificación", "Segmentación", "Lesión aislada", "Atención (Grad-CAM)", "Model card"]
)

with tab_cls:
    st.subheader("Clasificación en 7 clases")
    left, right = st.columns([1, 1])
    left.image(image, caption="Imagen de entrada", width="stretch")
    ordered = dict(sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True))
    right.bar_chart(ordered, horizontal=True)
    right.dataframe(
        [
            {
                "clase": name,
                "diagnóstico": pipeline.class_names_es.get(name, name),
                "probabilidad": f"{prob:.2%}",
            }
            for name, prob in ordered.items()
        ],
        hide_index=True,
        width="stretch",
    )

with tab_seg:
    st.subheader("Segmentación de la lesión")
    a, b = st.columns(2)
    a.image(result.segmentation.mask * 255, caption="Máscara binaria predicha", width="stretch", clamp=True)
    b.image(result.segmentation.overlay, caption="Contorno sobre la imagen original", width="stretch")

with tab_iso:
    st.subheader("Lesión extraída del fondo")
    if not detected:
        st.info("No se detectó una lesión: no hay región que aislar.")
    else:
        st.write(
            "La máscara se usa como canal alfa para eliminar la piel circundante. "
            "El recorte conserva un pequeño margen perilesional, que es relevante "
            "para la valoración clínica del borde."
        )
        a, b = st.columns(2)
        a.image(result.segmentation.rgba, caption="Lesión sobre fondo transparente", width="stretch")
        b.image(result.segmentation.cropped_rgba, caption="Recorte a la caja de la lesión", width="stretch")
        if result.segmentation.bbox:
            x, y, w, h = result.segmentation.bbox
            st.caption(f"Caja de la lesión: x={x}, y={y}, ancho={w}, alto={h} px")

with tab_attn:
    st.subheader("Atención e interpretabilidad")
    a, b = st.columns(2)
    a.image(result.gradcam_overlay, caption="Grad-CAM de la clase predicha", width="stretch")
    b.image(result.gradcam, caption="Mapa de Grad-CAM sin superponer", width="stretch", clamp=True)
    if detected:
        st.metric(
            "Masa de atención dentro de la lesión",
            f"{result.attention_in_lesion:.1%}",
            help=(
                "Fracción del mapa de Grad-CAM que cae dentro de la máscara PREDICHA por el "
                "segmentador. No es comparable con la tabla del informe, que usa la máscara "
                "anotada del dataset."
            ),
        )
        st.caption(
            "Un valor alto indica que el clasificador se apoyó en la lesión; uno bajo, que "
            "también usó la piel alrededor o artefactos como vello, burbujas o el viñeteado "
            "del dermatoscopio. Un valor bajo no implica por sí solo que la predicción sea "
            "incorrecta: el borde y la piel vecina también aportan información."
        )

with tab_card:
    st.subheader("Model card")
    card = REPO_ROOT / "MODEL_CARD.md"
    st.markdown(card.read_text(encoding="utf-8") if card.exists() else "No se encontró `MODEL_CARD.md`.")

st.divider()
st.caption(
    "Latencia de esta inferencia: "
    + " | ".join(f"{k} {v:.0f} ms" for k, v in result.latency_ms.items())
    + f" | total {sum(result.latency_ms.values()):.0f} ms"
)

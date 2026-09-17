"""Aplicativo Streamlit (§4.6).

El enunciado es explicito en que no se evalua el diseno visual, sino que el flujo
funcione de principio a fin sin errores. Asi que la interfaz es deliberadamente plana:
una fuente de imagen, un boton, y las cuatro salidas obligatorias en pestanas.

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

st.set_page_config(page_title="DermaScope", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos ...")
def get_pipeline() -> DermaPipeline:
    """Una sola carga de modelos por proceso.

    Sin `cache_resource`, Streamlit reejecuta el script completo en cada interaccion y
    recargaria el clasificador y la U-Net desde disco cada vez, lo que hace la demo inusable.
    """
    return DermaPipeline()


def read_image(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return np.array(image)


st.title("DermaScope — Analisis dermatoscopico asistido")
st.caption(
    "Herramienta academica de apoyo. **No es un dispositivo diagnostico** y no "
    "sustituye la valoracion de un dermatologo."
)

with st.sidebar:
    st.header("Entrada")
    source = st.radio("Fuente de la imagen", ["Subir archivo", "Camara"])
    file = (
        st.file_uploader("Imagen dermatoscopica", type=["jpg", "jpeg", "png"])
        if source == "Subir archivo"
        else st.camera_input("Captura")
    )
    smooth_samples = st.slider(
        "Muestras de SmoothGrad-CAM", 1, 32, 8,
        help="Mas muestras dan un mapa mas estable a costa de latencia.",
    )
    st.divider()
    st.caption(f"Dispositivo de inferencia: `{get_pipeline().device}`")

if file is None:
    st.info("Carga una imagen dermatoscopica para iniciar el analisis.")
    st.stop()

image = read_image(file)
pipeline = get_pipeline()

with st.spinner("Ejecutando clasificacion, segmentacion y atencion ..."):
    result = pipeline.run(image, smooth_samples=smooth_samples)

# --- Resumen -----------------------------------------------------------------
label_es = pipeline.class_names_es.get(result.predicted_class, result.predicted_class)
confidence = result.probabilities[result.predicted_class]

col1, col2, col3 = st.columns(3)
col1.metric("Clase predicha", result.predicted_class.upper(), label_es)
col2.metric("Confianza", f"{confidence:.1%}")
col3.metric(
    "Area de la lesion",
    f"{result.segmentation.area_ratio:.1%}",
    "de la imagen",
)

if not result.segmentation.detected:
    st.warning(
        "La segmentacion no encontro una region de lesion por encima del area minima. "
        "El aislamiento puede no ser confiable en esta imagen."
    )

tab_cls, tab_seg, tab_iso, tab_attn, tab_card = st.tabs(
    ["Clasificacion", "Segmentacion", "Lesion aislada", "Atencion", "Model card"]
)

with tab_cls:
    st.subheader("Clasificacion en 7 clases (§4.1)")
    left, right = st.columns([1, 1])
    left.image(image, caption="Imagen de entrada", use_container_width=True)
    ordered = dict(
        sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    )
    right.bar_chart(ordered, horizontal=True)
    right.dataframe(
        [
            {
                "clase": name,
                "diagnostico": pipeline.class_names_es.get(name, name),
                "probabilidad": f"{prob:.2%}",
            }
            for name, prob in ordered.items()
        ],
        hide_index=True,
        use_container_width=True,
    )

with tab_seg:
    st.subheader("Segmentacion de la lesion (§4.2)")
    a, b = st.columns(2)
    a.image(
        result.segmentation.mask * 255,
        caption="Mascara binaria predicha",
        use_container_width=True,
        clamp=True,
    )
    b.image(
        result.segmentation.overlay,
        caption="Contorno sobre la imagen original",
        use_container_width=True,
    )

with tab_iso:
    st.subheader("Lesion extraida del fondo (§4.2)")
    st.write(
        "La mascara se usa como canal alfa para eliminar la piel circundante. "
        "El recorte conserva un pequeno margen perilesional, que es relevante "
        "para la valoracion clinica del borde."
    )
    a, b = st.columns(2)
    a.image(
        result.segmentation.rgba,
        caption="Lesion sobre fondo transparente",
        use_container_width=True,
    )
    b.image(
        result.segmentation.cropped_rgba,
        caption="Recorte al bounding box de la lesion",
        use_container_width=True,
    )
    if result.segmentation.bbox:
        x, y, w, h = result.segmentation.bbox
        st.caption(f"Bounding box: x={x}, y={y}, ancho={w}, alto={h} px")

with tab_attn:
    st.subheader("Atencion e interpretabilidad (§4.3)")
    a, b = st.columns(2)
    a.image(
        result.gradcam_overlay,
        caption="SmoothGrad-CAM sobre la clase predicha",
        use_container_width=True,
    )
    b.image(
        result.gradcam,
        caption="Mapa de atencion crudo",
        use_container_width=True,
        clamp=True,
    )
    st.metric(
        "Masa de atencion dentro de la lesion",
        f"{result.attention_in_lesion:.1%}",
        help=(
            "Fraccion del mapa de SmoothGrad-CAM que cae dentro de la mascara PREDICHA por "
            "el segmentador. Un valor alto indica que el clasificador decidio mirando la "
            "lesion y no artefactos del fondo como vello, burbujas o el vineteado del "
            "dermatoscopio. No es comparable con la tabla del informe, que usa Grad-CAM y "
            "la mascara anotada del dataset."
        ),
    )
    if result.attention_in_lesion < 0.5:
        st.warning(
            "Menos de la mitad de la atencion cae sobre la lesion segmentada. "
            "La prediccion podria estar apoyandose en artefactos de la imagen."
        )

with tab_card:
    st.subheader("Model card (§4.7)")
    card = REPO_ROOT / "MODEL_CARD.md"
    st.markdown(
        card.read_text(encoding="utf-8")
        if card.exists()
        else "No se encontro `MODEL_CARD.md`."
    )

st.divider()
st.caption(
    "Latencia de esta inferencia: "
    + " | ".join(f"{k} {v:.0f} ms" for k, v in result.latency_ms.items())
    + f" | total {sum(result.latency_ms.values()):.0f} ms"
)

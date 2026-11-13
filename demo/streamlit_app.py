"""Demo del motor de decisión de crédito (credit-risk-agent-lab) en Streamlit Community Cloud.

Corre solo el código: PD del campeón, economía del préstamo, política de crédito, razones TreeSHAP y la
carta para el cliente. No llama a ningún LLM; el agente con Claude vive en el repositorio.
Los modelos se descargan la primera vez del repo de modelos en Hugging Face.

En local, desde la raíz del repo:  streamlit run demo/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DEMO = Path(__file__).resolve().parent
RAIZ = DEMO.parent
sys.path.insert(0, str(RAIZ / "src"))

REPO_MODELO = "neiderolaya26/credit-risk-lending-club"
ARCHIVOS = ["campeon.joblib", "prepago_c_supervivencia.joblib", "severidad_b_economica.joblib", "referencia_tasa.joblib"]
REGLAS = {"1": "Regla 1: ROI esperado ≥ 4% y cuota ≤ 20% del ingreso",
          "2b": "Regla 2b: se baja el monto hasta que cumple",
          "2a": "Regla 2a: se sube la tasa (máximo +2 pts, dentro de la banda de mercado)",
          "3": "Regla 3: ninguna oferta cumple"}

st.set_page_config(page_title="Motor de decisión de crédito", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Descargando los modelos de Hugging Face…")
def herramientas():
    from huggingface_hub import hf_hub_download

    destino = RAIZ / "modelos"  # el motor lee los modelos de <raíz del repo>/modelos
    destino.mkdir(exist_ok=True)
    for nombre in ARCHIVOS:
        if not (destino / nombre).exists():
            hf_hub_download(REPO_MODELO, nombre, local_dir=destino)
    from credit_lab.agente import Herramientas

    return Herramientas()


@st.cache_data
def ejemplos() -> pd.DataFrame:
    return pd.read_csv(DEMO / "ejemplos.csv", parse_dates=["issue_d"])


HERR, EJ = herramientas(), ejemplos()

st.title("🏦 Motor de decisión de crédito · Lending Club")
st.markdown(
    "Elige una solicitud real de jul–dic 2015 que el modelo **nunca vio**, o cambia sus datos, y mira qué decide "
    "la política. Todo lo calcula el código del repositorio: PD, economía, política y razones. **Sin LLM.**  \n"
    "[Repositorio](https://github.com/neiderolaya26/credit-risk-agent-lab) · "
    f"[Modelo y ficha en Hugging Face](https://huggingface.co/{REPO_MODELO})")

etiquetas = [f"#{i} · {f.sub_grade} · USD {f.loan_amnt:,.0f} · {f.purpose}" for i, f in EJ.iterrows()]
i = st.selectbox("Solicitud de ejemplo", range(len(EJ)), format_func=lambda k: etiquetas[k])
original = EJ.iloc[[i]]
f = original.iloc[0]

with st.form("solicitud"):
    c1, c2, c3 = st.columns(3)
    monto = c1.number_input("Monto pedido (USD)", 1000, 35000, int(f.loan_amnt), step=500, key=f"m{i}")
    tasa = c2.number_input("Tasa ofrecida por Lending Club (%)", 5.0, 30.0, float(f.int_rate), step=0.25, key=f"t{i}")
    ingreso = c3.number_input("Ingreso anual (USD)", 1000, 2_000_000, int(f.annual_inc), step=1000, key=f"i{i}")
    c4, c5, c6, c7 = st.columns(4)
    dti = c4.number_input("Deuda / ingreso (%)", 0.0, 100.0, float(f.dti), key=f"d{i}")
    fico = c5.number_input("FICO (límite inferior)", 600, 850, int(f.fico_range_low), step=5, key=f"f{i}")
    uso = c6.number_input("Uso del crédito rotativo (%)", 0.0, 150.0,
                          float(0 if pd.isna(f.revol_util) else f.revol_util), key=f"u{i}")
    consultas = c7.number_input("Consultas de crédito (6 meses)", 0, 30, int(f.inq_last_6mths), key=f"c{i}")
    st.form_submit_button("Evaluar", type="primary")

sol = original.copy()
cambios = {"loan_amnt": monto, "funded_amnt": monto, "int_rate": tasa, "annual_inc": ingreso, "dti": dti,
           "fico_range_low": fico, "fico_range_high": fico + 4, "revol_util": uso, "inq_last_6mths": consultas}
for k, v in cambios.items():
    sol[k] = v
sol = sol.astype(original.dtypes.to_dict())
modificada = any(float(sol[k].iloc[0]) != float(original[k].iloc[0]) for k in cambios if pd.notna(original[k].iloc[0]))

pol, riesgo, eco = HERR.politica(sol), HERR.riesgo(sol), HERR.economia(sol)
decision = pol["decision"]

color = {"aprobar": "green", "contraoferta": "orange", "rechazar": "red"}[decision]
st.subheader(f"Decisión de la política: :{color}[{decision}]")
oferta = f"USD {pol['monto']:,.0f} a {pol['tasa']:.2f}%" if decision != "rechazar" else "sin oferta"
st.caption(f"{REGLAS[pol['regla']]} · Oferta: {oferta}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("PD del campeón", f"{riesgo['pd']:.1%}",
          f"{(riesgo['pd'] - riesgo['pd_promedio_cartera']) * 100:+.1f} pts vs. cartera", delta_color="inverse")
m2.metric("ROI esperado a la tasa pedida", f"{eco['roi_esperado']:.1%}")
m3.metric("Tasa de mercado del perfil", f"{pol['tasa_mercado']:.2f}%")
m4.metric("Tasa sugerida (ROI ≥ 4%)", "—" if pol["tasa_sugerida"] is None else f"{pol['tasa_sugerida']:.2f}%")

if modificada:
    st.info("Solicitud modificada: el resultado real ya no aplica.")
else:
    real = "terminó en mora" if f.malo else "se pagó"
    st.info(f"Resultado real de este préstamo (el modelo no lo vio): **{real}**.")

izq, der = st.columns([3, 2])
with izq:
    st.markdown("#### Razones del riesgo para el analista")
    st.caption("TreeSHAP en escala logit; positivo sube la PD. Sin proxies de edad.")
    razones = pd.DataFrame(riesgo["razones"]).rename(columns={
        "descripcion": "Variable", "valor": "Valor", "efecto": "Efecto", "contribucion_logit": "Contribución (logit)"})
    razones["Valor"] = razones["Valor"].map(lambda v: f"{v:,.2f}" if isinstance(v, float) else str(v))
    st.dataframe(razones, hide_index=True, width="stretch")
with der:
    st.markdown("#### Aviso para el cliente")
    st.caption("Catálogo determinista, máximo 4 razones.")
    if decision in {"rechazar", "contraoferta"}:
        from credit_lab.razones import carta

        st.text(carta(decision, HERR.razones_cliente(sol), pol))
    else:
        st.success("Aprobada: no requiere aviso al cliente.")

st.divider()
st.caption("En el repositorio, un agente LangGraph con Claude analiza el caso, opina y redacta el memo, y un analista "
           "firma lo que no es una aprobación clara. Datos públicos de Lending Club; no es un sistema para decidir "
           "créditos reales.")

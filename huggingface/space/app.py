"""Demo del motor de decisión de crédito (credit-risk-agent-lab) en un Space de Hugging Face.

Corre solo el código: PD del campeón, economía del préstamo, política de crédito, razones TreeSHAP y la
carta para el cliente. No llama a ningún LLM; el agente con Claude vive en el repositorio.
Los modelos se descargan del repo de modelos en el Hub la primera vez.
"""

import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))  # el paquete credit_lab del repositorio, copiado en src/
REPO_MODELOS = os.environ.get("REPO_MODELOS", "neiderolaya26/credit-risk-lending-club")
ARCHIVOS = ["campeon.joblib", "prepago_c_supervivencia.joblib", "severidad_b_economica.joblib", "referencia_tasa.joblib"]


def _descargar_modelos() -> None:
    from huggingface_hub import hf_hub_download

    destino = RAIZ / "modelos"
    destino.mkdir(exist_ok=True)
    for nombre in ARCHIVOS:
        if not (destino / nombre).exists():
            hf_hub_download(REPO_MODELOS, nombre, local_dir=destino)


_descargar_modelos()

from credit_lab.agente import Herramientas  # noqa: E402  (después de descargar los modelos)
from credit_lab.razones import carta  # noqa: E402

HERR = Herramientas()
EJEMPLOS = pd.read_csv(RAIZ / "ejemplos.csv", parse_dates=["issue_d"])
PROPOSITOS = sorted(EJEMPLOS["purpose"].dropna().unique())
VIVIENDAS = sorted(EJEMPLOS["home_ownership"].dropna().unique())


def _etiqueta(i: int, f) -> str:
    return f"#{i} · {f['sub_grade']} · USD {f['loan_amnt']:,.0f} · {f['purpose']}"


OPCIONES = [_etiqueta(i, f) for i, f in EJEMPLOS.iterrows()]


def cargar(opcion: str):
    f = EJEMPLOS.iloc[OPCIONES.index(opcion)]
    return (f["loan_amnt"], f["int_rate"], f["annual_inc"], f["dti"], f["fico_range_low"], f["revol_util"],
            f["inq_last_6mths"], f["purpose"], f["home_ownership"])


def evaluar(opcion, monto, tasa, ingreso, dti, fico, uso_rotativo, consultas, proposito, vivienda):
    i = OPCIONES.index(opcion)
    original = EJEMPLOS.iloc[[i]].copy()
    sol = original.copy()
    cambios = {"loan_amnt": monto, "int_rate": tasa, "annual_inc": ingreso, "dti": dti, "fico_range_low": fico,
               "fico_range_high": fico + 4, "revol_util": uso_rotativo, "inq_last_6mths": consultas,
               "purpose": proposito, "home_ownership": vivienda}
    for k, v in cambios.items():
        sol[k] = v
    sol["funded_amnt"] = sol["loan_amnt"]
    sol = sol.astype(original.dtypes.to_dict())
    modificada = any(float(sol[k].iloc[0]) != float(original[k].iloc[0]) for k in
                     ("loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low", "revol_util", "inq_last_6mths")) \
        or sol["purpose"].iloc[0] != original["purpose"].iloc[0] or sol["home_ownership"].iloc[0] != original["home_ownership"].iloc[0]

    pol = HERR.politica(sol)
    riesgo = HERR.riesgo(sol)
    eco = HERR.economia(sol)
    decision = pol["decision"]
    regla = {"1": "regla 1: ROI ≥ 4% y cuota ≤ 20% del ingreso", "2b": "regla 2b: se baja el monto",
             "2a": "regla 2a: se sube la tasa (máximo +2 pts)", "3": "regla 3: ninguna oferta cumple"}[pol["regla"]]
    oferta = (f"USD {pol['monto']:,.0f} a {pol['tasa']:.2f}%" if decision != "rechazar" else "sin oferta")
    sugerida = "—" if pol["tasa_sugerida"] is None else f"{pol['tasa_sugerida']:.2f}%"
    resumen = (f"### Decisión de la política: **{decision}**\n{regla} · oferta: {oferta}\n\n"
               f"| Métrica | Valor |\n|---|---|\n"
               f"| PD del campeón (LightGBM recalibrado) | {riesgo['pd']:.1%} |\n"
               f"| PD promedio de la cartera | {riesgo['pd_promedio_cartera']:.1%} |\n"
               f"| ROI esperado a la tasa pedida | {eco['roi_esperado']:.1%} |\n"
               f"| Tasa de mercado del perfil | {pol['tasa_mercado']:.2f}% |\n"
               f"| Tasa sugerida (ROI ≥ 4%) | {sugerida} |\n")
    if not modificada:
        resumen += f"\nResultado real de este préstamo (el modelo no lo vio): **{'mora' if original['malo'].iloc[0] else 'pagó'}**"
    else:
        resumen += "\nSolicitud modificada: el resultado real ya no aplica."
    razones = pd.DataFrame(riesgo["razones"]).rename(columns={
        "descripcion": "Variable", "valor": "Valor", "efecto": "Efecto", "contribucion_logit": "Contribución (logit)"})
    texto_carta = ""
    if decision in {"rechazar", "contraoferta"}:
        texto_carta = carta(decision, HERR.razones_cliente(sol), pol)
    return resumen, razones, texto_carta or "Aprobada: no requiere aviso al cliente."


with gr.Blocks(title="Motor de decisión de crédito") as demo:
    gr.Markdown(
        "# Motor de decisión de crédito · Lending Club\n"
        "Elige una solicitud real de jul–dic 2015 (el modelo no la vio) o cambia sus datos. Todo lo calcula el "
        "código del repo: PD, economía, política y razones. Sin LLM.\n\n"
        "[Repositorio](https://github.com/neiderolaya26/credit-risk-agent-lab) · "
        "[Modelo y ficha](https://huggingface.co/" + REPO_MODELOS + ")")
    ejemplo = gr.Dropdown(OPCIONES, value=OPCIONES[0], label="Solicitud de ejemplo")
    with gr.Row():
        monto = gr.Number(label="Monto pedido (USD)", minimum=1000, maximum=35000)
        tasa = gr.Number(label="Tasa ofrecida por Lending Club (%)", minimum=5, maximum=30)
        ingreso = gr.Number(label="Ingreso anual (USD)", minimum=1000)
    with gr.Row():
        dti = gr.Number(label="Deuda / ingreso (%)")
        fico = gr.Number(label="FICO (límite inferior del rango)", minimum=600, maximum=850)
        uso = gr.Number(label="Uso del crédito rotativo (%)")
        consultas = gr.Number(label="Consultas de crédito (6 meses)", minimum=0)
    with gr.Row():
        proposito = gr.Dropdown(PROPOSITOS, label="Propósito")
        vivienda = gr.Dropdown(VIVIENDAS, label="Vivienda")
    boton = gr.Button("Evaluar", variant="primary")
    salida = gr.Markdown()
    gr.Markdown("#### Razones del riesgo para el analista (TreeSHAP, sin proxies de edad)")
    tabla = gr.Dataframe()
    gr.Markdown("#### Aviso para el cliente (catálogo determinista, máximo 4 razones)")
    aviso = gr.Textbox(lines=8, show_label=False)
    campos = [monto, tasa, ingreso, dti, fico, uso, consultas, proposito, vivienda]
    ejemplo.change(cargar, ejemplo, campos)
    boton.click(evaluar, [ejemplo, *campos], [salida, tabla, aviso])
    demo.load(cargar, ejemplo, campos)

if __name__ == "__main__":
    demo.launch()

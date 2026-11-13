"""Razones para el cliente: catálogo determinista, separado del memo interno (decisión 023, issue #37).

Un rechazo, o una contraoferta que cambia las condiciones, debe explicarse al cliente con razones
consistentes que no dependan de cómo redacte un LLM. Aquí las razones salen de TreeSHAP:
    1. se toman las variables que SUBEN el riesgo, ordenadas por contribución;
    2. se descartan las que no se pueden dar como razón (proxies de edad, ubicación, vivienda);
    3. se mapean a un texto aprobado del catálogo, sin cifras internas del modelo;
    4. máximo 4 por caso, como suele exigir un aviso de acción adversa.

El LLM puede redactar el memo para el analista, pero no elige ni redacta estas razones.
"""

import pandas as pd

MAX_RAZONES = 4

# proxies de atributos protegidos (issue #39): la antigüedad del historial crece con la edad
PROXIES_EDAD = {"antig_credito_meses", "mo_sin_old_rev_tl_op"}
# variables que el modelo usa pero que no son una razón defendible ante el cliente
NO_RAZON_CLIENTE = PROXIES_EDAD | {"addr_state", "home_ownership"}

CATALOGO = {
    "fico_medio": "Su puntaje de crédito es bajo para las condiciones solicitadas.",
    "dti": "Su nivel de endeudamiento frente a su ingreso es alto.",
    "inq_last_6mths": "Tiene varias solicitudes de crédito recientes.",
    "monto_ingreso": "El monto solicitado es alto frente a su ingreso.",
    "annual_inc": "Su ingreso es insuficiente para el monto solicitado.",
    "num_actv_rev_tl": "Tiene muchas cuentas rotativas activas.",
    "total_bc_limit": "El cupo total de sus tarjetas es bajo.",
    "purpose": "El propósito del préstamo tiene un riesgo mayor en nuestra experiencia.",
    "percent_bc_gt_75": "Una proporción alta de sus tarjetas tiene un uso mayor al 75% del cupo.",
    "pct_tl_nvr_dlq": "Su historial de pagos registra cuentas con atrasos.",
    "avg_cur_bal": "El saldo de sus cuentas es alto frente a su perfil.",
    "tot_cur_bal": "El saldo total de sus deudas es alto.",
    "total_rev_hi_lim": "El cupo total de sus créditos rotativos es bajo.",
    "mort_acc": "Su historial de créditos hipotecarios es limitado.",
    "bc_util": "El uso de sus tarjetas de crédito es alto.",
    "revol_util": "El uso de sus créditos rotativos es alto.",
    "revol_bal": "El saldo de sus créditos rotativos es alto.",
    "emp_length": "Su antigüedad laboral es corta.",
    "total_acc": "Su número de cuentas de crédito no favorece la solicitud.",
    "loan_amnt": "El monto solicitado es alto para su perfil.",
    "mths_since_last_delinq": "Tiene atrasos de pago recientes.",
    "delinq_2yrs": "Tiene atrasos de pago en los últimos 2 años.",
    "pub_rec": "Tiene registros públicos negativos.",
    "pub_rec_bankruptcies": "Tiene quiebras registradas.",
    "num_tl_90g_dpd_24m": "Tiene cuentas con atrasos de 90 días o más en los últimos 24 meses.",
    "open_acc": "Su número de cuentas abiertas no favorece la solicitud.",
    "acc_now_delinq": "Tiene cuentas actualmente en mora.",
}


def razones_cliente(contribuciones: pd.Series, maximo: int = MAX_RAZONES) -> list[dict]:
    """Hasta `maximo` razones del catálogo, de la que más sube el riesgo a la que menos.
    `contribuciones`: contribución SHAP (logit) por variable de un préstamo, sin el valor base."""
    suben = contribuciones[contribuciones > 0].sort_values(ascending=False)
    out = []
    for variable in suben.index:
        if variable in NO_RAZON_CLIENTE or variable not in CATALOGO:
            continue
        out.append({"variable": variable, "texto": CATALOGO[variable]})
        if len(out) == maximo:
            break
    return out


def carta(decision: str, razones: list[dict], oferta: dict | None = None) -> str:
    """Plantilla del aviso al cliente (acción adversa o precio basado en riesgo), separada del memo interno."""
    lista = "\n".join(f"{i}. {r['texto']}" for i, r in enumerate(razones, 1)) or "1. No cumple la política de crédito vigente."
    if decision == "rechazar":
        cuerpo = "Después de revisar su solicitud, no podemos aprobar el crédito en este momento."
    elif decision == "contraoferta" and oferta:
        cuerpo = (f"No podemos aprobar el crédito en las condiciones solicitadas. Podemos ofrecerle "
                  f"USD {oferta['monto']:,.0f} a una tasa anual de {oferta['tasa']:.2f}%.")
    else:
        return ""
    return (f"{cuerpo}\n\nLas razones principales son:\n{lista}\n\n"
            "Tiene derecho a conocer la información de crédito usada en esta decisión y a pedir su revisión.")

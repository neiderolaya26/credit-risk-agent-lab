"""Explicabilidad del campeón con SHAP (TreeSHAP nativo de LightGBM).

Para cada préstamo, LightGBM descompone su logit en:
    logit = valor_base + suma(contribución de cada variable)
La recalibración (decisión 007) solo suma una constante, así que:
    PD = sigmoide(valor_base + desplazamiento + suma(contribuciones))

Contribución positiva = sube el riesgo; negativa = lo baja.
"""

import numpy as np
import pandas as pd

from credit_lab.campeon import Campeon
from credit_lab.features import ARBOLES, derivar

NOMBRES = {
    "fico_medio": "Puntaje FICO",
    "dti": "Endeudamiento sobre ingreso (DTI)",
    "inq_last_6mths": "Consultas de crédito en los últimos 6 meses",
    "monto_ingreso": "Monto solicitado sobre ingreso anual",
    "annual_inc": "Ingreso anual",
    "num_actv_rev_tl": "Cuentas rotativas activas",
    "total_bc_limit": "Cupo total en tarjetas",
    "purpose": "Propósito del préstamo",
    "mo_sin_old_rev_tl_op": "Antigüedad de la cuenta rotativa más vieja (meses)",
    "percent_bc_gt_75": "% de tarjetas con uso mayor al 75%",
    "home_ownership": "Tipo de vivienda",
    "pct_tl_nvr_dlq": "% de cuentas que nunca han estado en mora",
    "avg_cur_bal": "Saldo promedio de las cuentas",
    "tot_cur_bal": "Saldo total de las cuentas",
    "total_rev_hi_lim": "Cupo total rotativo",
    "mort_acc": "Créditos hipotecarios",
    "bc_util": "Uso de tarjetas (%)",
    "antig_credito_meses": "Antigüedad crediticia (meses)",
    "revol_util": "Uso del crédito rotativo (%)",
    "revol_bal": "Saldo rotativo",
    "emp_length": "Antigüedad laboral",
    "addr_state": "Estado de residencia",
    "total_acc": "Total de cuentas de crédito",
    "loan_amnt": "Monto solicitado",
    "mths_since_last_delinq": "Meses desde la última mora",
    "delinq_2yrs": "Moras en los últimos 2 años",
    "pub_rec": "Registros públicos negativos",
    "pub_rec_bankruptcies": "Quiebras registradas",
    "num_tl_90g_dpd_24m": "Cuentas con mora de 90+ días en 24 meses",
    "open_acc": "Cuentas abiertas",
    "acc_now_delinq": "Cuentas actualmente en mora",
}


def _sigmoide(z):
    return 1 / (1 + np.exp(-z))


def contribuciones(campeon: Campeon, df: pd.DataFrame) -> pd.DataFrame:
    """Contribución SHAP (en logit) de cada variable para cada préstamo, más el valor base."""
    X = campeon.pre.transform(derivar(df)[ARBOLES])
    c = campeon.modelo.booster_.predict(X, pred_contrib=True)
    out = pd.DataFrame(c[:, :-1], columns=X.columns, index=df.index)
    out["valor_base"] = c[:, -1] + campeon.desplazamiento
    return out


def explicar_prestamo(campeon: Campeon, fila: pd.DataFrame, top: int = 5) -> dict:
    """PD recalibrada y principales razones de un préstamo (una fila)."""
    contrib = contribuciones(campeon, fila).iloc[0]
    base = contrib.pop("valor_base")
    datos = derivar(fila).iloc[0]
    orden = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    razones = [{
        "variable": v,
        "descripcion": NOMBRES.get(v, v),
        "valor": datos.get(v),
        "contribucion_logit": float(orden[v]),
        "efecto": "sube el riesgo" if orden[v] > 0 else "baja el riesgo",
    } for v in orden.index[:top]]
    return {"pd": float(_sigmoide(base + contrib.sum())), "pd_base": float(_sigmoide(base)),
            "razones": razones}

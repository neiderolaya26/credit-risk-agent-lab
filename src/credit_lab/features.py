"""Variables del modelo según la decisión 002 (docs/decisiones.md).

- Regresión logística: variables limpias con IV >= 0,02 en entrenamiento.
- Modelos de árboles y red neuronal: todas las variables limpias.

"Limpia" = sin fuga de información, sin redundancia y sin reflejar la
política de Lending Club (ver notebooks/02_eda_variables.ipynb).
"""

import pandas as pd

# Excluidas y por qué
EXCLUIDAS = {
    "installment": "se calcula con int_rate (score de Lending Club)",
    "funded_amnt": "monto decidido por los inversionistas; duplica loan_amnt",
    "fico_range_low": "redundante: se usa fico_medio",
    "fico_range_high": "redundante: se usa fico_medio",
    "initial_list_status": "decisión operativa de Lending Club",
    "verification_status": "política de Lending Club",
    "application_type": "constante en la población (todas Individual)",
}

LOGISTICA = [
    "fico_medio", "annual_inc", "total_bc_limit", "avg_cur_bal", "tot_cur_bal",
    "total_rev_hi_lim", "dti", "monto_ingreso", "home_ownership", "mort_acc",
    "percent_bc_gt_75", "bc_util", "mo_sin_old_rev_tl_op", "purpose",
    "antig_credito_meses", "num_actv_rev_tl",
]

ARBOLES = LOGISTICA + [
    "revol_util", "inq_last_6mths", "revol_bal", "emp_length",
    "addr_state", "total_acc", "loan_amnt", "mths_since_last_delinq",
    "delinq_2yrs", "pct_tl_nvr_dlq", "pub_rec", "pub_rec_bankruptcies",
    "num_tl_90g_dpd_24m", "open_acc", "acc_now_delinq",
]

CATEGORICAS = ["home_ownership", "purpose", "emp_length", "addr_state"]


def derivar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables derivadas disponibles al momento de la solicitud."""
    df = df.copy()
    df["fico_medio"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["antig_credito_meses"] = (
        df["issue_d"] - pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    ).dt.days / 30.4
    df["monto_ingreso"] = df["loan_amnt"] / df["annual_inc"]
    return df

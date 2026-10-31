"""Población de modelamiento según la decisión 001 (docs/decisiones.md).

- Préstamos a 36 meses originados entre sep-2012 y dic-2015 (decisión 003:
  antes de sep-2012 Lending Club no reportaba varias variables de buró).
- Malo = Charged Off / Default; bueno = Fully Paid.
- Entrenamiento sep-2012 a 2014; validación fuera de tiempo (OOT) 2015.
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PARQUET = RAIZ / "data" / "interim" / "accepted.parquet"

ESTADO_MALO = {"Charged Off", "Default"}
ESTADO_BUENO = {"Fully Paid"}
ANIOS = (2012, 2015)
INICIO = pd.Timestamp("2012-09-01")
ANIO_OOT = 2015

# Variables que no pueden ser features: ocurren después de la originación
# o son la decisión de riesgo de Lending Club (benchmark).
FUGA = [
    "total_pymnt", "recoveries", "collection_recovery_fee", "out_prncp", "last_pymnt_d",
    "last_fico_range_high", "hardship_flag", "debt_settlement_flag",
]
BENCHMARK = ["int_rate", "grade", "sub_grade"]


def cargar_poblacion(ruta: Path = PARQUET, inicio: pd.Timestamp = INICIO) -> pd.DataFrame:
    """`inicio` solo se cambia para reproducir el diagnóstico de la decisión 003."""
    df = pd.read_parquet(ruta)
    df["term"] = df["term"].str.strip()
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["anio"] = df["issue_d"].dt.year

    filtro = (
        df["term"].eq("36 months")
        & df["anio"].between(*ANIOS)
        & df["issue_d"].ge(inicio)
        & df["loan_status"].isin(ESTADO_MALO | ESTADO_BUENO)
    )
    df = df.loc[filtro].copy()
    df["malo"] = df["loan_status"].isin(ESTADO_MALO).astype("int8")
    df["muestra"] = df["anio"].map(lambda a: "oot" if a == ANIO_OOT else "train")
    return df.reset_index(drop=True)

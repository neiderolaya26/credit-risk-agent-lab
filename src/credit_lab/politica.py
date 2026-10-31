"""Política de aprobación por utilidad esperada.

Para cada préstamo, con su subgrado (el precio que ofrece Lending Club):
    r_bueno = utilidad / monto promedio de los préstamos buenos del subgrado
    r_malo  = utilidad / monto promedio de los préstamos malos del subgrado
Ambos se estiman con el entrenamiento (sep-2012 a 2014), con flujos reales.

    utilidad_esperada = monto × [(1 − PD) × r_bueno + PD × r_malo]
    Aprobar si utilidad_esperada > margen × monto

La PD de equilibrio de un subgrado es r_bueno / (r_bueno − r_malo): con
precios más altos se toleran PD más altas. Por eso un corte único de PD
no tiene sentido económico.
"""

import numpy as np
import pandas as pd

from credit_lab.negocio import utilidad_prestamo
from credit_lab.poblacion import RAIZ

RUTA_RENDIMIENTOS = RAIZ / "data" / "processed" / "rendimientos_subgrado.csv"


def cargar_rendimientos() -> pd.DataFrame:
    """Rendimientos por subgrado ya calculados (decisión 008), sin cargar los datos crudos."""
    return pd.read_csv(RUTA_RENDIMIENTOS, index_col="sub_grade")


def guardar_rendimientos() -> pd.DataFrame:
    from credit_lab.poblacion import cargar_poblacion
    df = cargar_poblacion()
    r = rendimientos_por_subgrado(df[df["muestra"].eq("train")])
    r["prestamos"] = df[df["muestra"].eq("train")]["sub_grade"].value_counts()
    RUTA_RENDIMIENTOS.parent.mkdir(parents=True, exist_ok=True)
    r.round(6).to_csv(RUTA_RENDIMIENTOS)
    return r


def rendimientos_por_subgrado(train: pd.DataFrame) -> pd.DataFrame:
    r = utilidad_prestamo(train) / train["funded_amnt"]
    t = (pd.DataFrame({"sub_grade": train["sub_grade"], "malo": train["malo"], "r": r})
         .groupby(["sub_grade", "malo"])["r"].mean().unstack())
    t.columns = ["r_bueno", "r_malo"]
    t["pd_equilibrio"] = t["r_bueno"] / (t["r_bueno"] - t["r_malo"])
    return t


def utilidad_esperada(df: pd.DataFrame, pd_estimada, rendimientos: pd.DataFrame) -> np.ndarray:
    r = rendimientos.reindex(df["sub_grade"].to_numpy())
    p = np.asarray(pd_estimada, dtype=float)
    return df["funded_amnt"].to_numpy() * ((1 - p) * r["r_bueno"].to_numpy() + p * r["r_malo"].to_numpy())


def aprobar(df: pd.DataFrame, pd_estimada, rendimientos: pd.DataFrame, margen: float = 0.0) -> np.ndarray:
    return utilidad_esperada(df, pd_estimada, rendimientos) > margen * df["funded_amnt"].to_numpy()

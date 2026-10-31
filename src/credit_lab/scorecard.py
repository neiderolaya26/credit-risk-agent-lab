"""Scorecard: regresión logística sobre WoE convertida a puntos.

Escala estándar de la industria (configurable):
- 600 puntos = odds de 50 buenos por cada malo.
- Cada 20 puntos (PDO, "points to double the odds") las odds se duplican.

Más puntos = menos riesgo.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_lab.transformadores import preprocesador

BASE, ODDS_BASE, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE - FACTOR * np.log(ODDS_BASE)


def modelo_logistica() -> Pipeline:
    """Pipeline completo: limpieza -> winsorización -> WoE -> logística."""
    pasos = preprocesador("logistica").steps
    return Pipeline(pasos + [("logistica", LogisticRegression(max_iter=2000))])


def pd_a_puntos(pd_malo) -> np.ndarray:
    pd_malo = np.clip(np.asarray(pd_malo, dtype=float), 1e-6, 1 - 1e-6)
    return OFFSET + FACTOR * np.log((1 - pd_malo) / pd_malo)


def tabla_puntos(modelo: Pipeline) -> pd.DataFrame:
    """Puntos que aporta cada tramo de cada variable. El intercepto se reparte por igual."""
    woe = modelo.named_steps["woe"]
    lr = modelo.named_steps["logistica"]
    coefs = dict(zip(woe.get_feature_names_out(), lr.coef_[0]))
    n = len(coefs)
    filas = []
    for var, b in coefs.items():
        for tramo, w in woe.woe_[var].items():
            puntos = -(b * w + lr.intercept_[0] / n) * FACTOR + OFFSET / n
            filas.append({"variable": var, "tramo": tramo, "woe": w, "puntos": round(puntos),
                          "orden": _inicio_tramo(tramo)})
    t = pd.DataFrame(filas)
    t["pos"] = t.groupby("variable", sort=False).ngroup()
    return t.sort_values(["pos", "orden", "tramo"]).drop(columns=["orden", "pos"]).reset_index(drop=True)


def _inicio_tramo(tramo: str) -> float:
    """Límite inferior de un tramo "(a, b]" para ordenarlo; categorías y NULO van al final."""
    try:
        return float(tramo.strip("([").split(",")[0])
    except ValueError:
        return np.inf

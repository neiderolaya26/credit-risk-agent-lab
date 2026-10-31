"""Weight of Evidence (WoE) e Information Value (IV).

WoE_i = ln(%buenos_i / %malos_i)
IV    = sum((%buenos_i - %malos_i) * WoE_i)

Guía usual del IV: < 0,02 inútil · 0,02–0,1 débil · 0,1–0,3 medio ·
0,3–0,5 fuerte · > 0,5 sospechoso (posible fuga de información).
"""

import numpy as np
import pandas as pd

SUAVIZADO = 0.5  # evita log(0) en tramos sin buenos o sin malos


def tramos(x: pd.Series, n: int = 10) -> pd.Series:
    """Tramos por cuantiles para numéricas; categorías tal cual. Nulos = tramo propio.

    Devuelve un categórico ordenado para que los tramos numéricos conserven
    su orden (con texto, "(7.0, 9.9]" quedaría después de "(28.0, 39.9]").
    """
    if pd.api.types.is_numeric_dtype(x):
        t = pd.qcut(x, q=n, duplicates="drop")
        t = t.cat.rename_categories([str(c) for c in t.cat.categories])
    else:
        t = x.astype("string").astype("category")
    return t.cat.add_categories("NULO").fillna("NULO")


def tabla_woe(x: pd.Series, y: pd.Series, n: int = 10) -> pd.DataFrame:
    t = tramos(x, n)
    g = pd.crosstab(t, y, dropna=False).reindex(columns=[0, 1], fill_value=0)
    g = g[g.sum(axis=1) > 0]
    g.index = g.index.astype(str)
    g.columns = ["buenos", "malos"]
    g["n"] = g["buenos"] + g["malos"]
    g["tasa_malo"] = g["malos"] / g["n"]
    pb = (g["buenos"] + SUAVIZADO) / (g["buenos"].sum() + SUAVIZADO)
    pm = (g["malos"] + SUAVIZADO) / (g["malos"].sum() + SUAVIZADO)
    g["woe"] = np.log(pb / pm)
    g["iv"] = (pb - pm) * g["woe"]
    return g


def information_value(x: pd.Series, y: pd.Series, n: int = 10) -> float:
    return float(tabla_woe(x, y, n)["iv"].sum())


def ranking_iv(df: pd.DataFrame, columnas: list[str], objetivo: str = "malo") -> pd.DataFrame:
    filas = [(c, information_value(df[c], df[objetivo])) for c in columnas]
    r = pd.DataFrame(filas, columns=["variable", "iv"]).sort_values("iv", ascending=False)
    r["fuerza"] = pd.cut(
        r["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
        labels=["inútil", "débil", "media", "fuerte", "sospechosa"],
    )
    return r.reset_index(drop=True)

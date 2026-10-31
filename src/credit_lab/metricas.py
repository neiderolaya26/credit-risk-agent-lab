"""Métricas de riesgo reutilizables."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score


def discriminacion(y, score) -> dict:
    """AUC, Gini y KS. `score` debe crecer con el riesgo (PD, tasa, etc.)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    auc = roc_auc_score(y, score)
    ks = ks_2samp(score[y == 1], score[y == 0]).statistic
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(ks)}


def resumen(y, pd_estimada) -> dict:
    """Discriminación + calibración para una PD (probabilidad entre 0 y 1)."""
    y = np.asarray(y)
    return {
        **discriminacion(y, pd_estimada),
        "brier": brier_score_loss(y, pd_estimada),
        "pd_media": float(np.mean(pd_estimada)),
        "tasa_real": float(y.mean()),
    }


def tabla_eficiencia(y, score, n: int = 10) -> pd.DataFrame:
    """Tabla de eficiencia (ganancias) por decil. Decil 1 = MAYOR riesgo.

    `score` debe crecer con el riesgo. Los acumulados van del decil más
    riesgoso al menos riesgoso: "% malos acum." responde "si rechazo los
    primeros k deciles, ¿qué parte de los malos evito?".
    """
    y = pd.Series(np.asarray(y), name="malo")
    rango = pd.Series(np.asarray(score, dtype=float)).rank(method="first", ascending=False)
    decil = pd.qcut(rango, n, labels=range(1, n + 1))
    t = y.groupby(decil.values).agg(prestamos="size", malos="sum")
    t["buenos"] = t["prestamos"] - t["malos"]
    t["tasa_malo"] = t["malos"] / t["prestamos"]
    t["%_malos"] = t["malos"] / t["malos"].sum()
    t["%_malos_acum"] = t["%_malos"].cumsum()
    t["%_buenos_acum"] = (t["buenos"] / t["buenos"].sum()).cumsum()
    t["ks"] = t["%_malos_acum"] - t["%_buenos_acum"]
    t["lift"] = t["tasa_malo"] / y.mean()
    t.index.name = "decil"
    return t


def resumen_orden(t: pd.DataFrame) -> dict:
    """Qué tan bien ordena un modelo, a partir de su tabla de eficiencia."""
    tasas = t["tasa_malo"].to_numpy()
    return {
        "inversiones": int((np.diff(tasas) > 0).sum()),  # decil menos riesgoso con MÁS mora
        "tasa_decil_1": tasas[0],
        "tasa_decil_10": tasas[-1],
        "veces_d1_vs_d10": tasas[0] / tasas[-1],
        "malos_capturados_top10%": t["%_malos_acum"].iloc[0],
        "malos_capturados_top20%": t["%_malos_acum"].iloc[1],
        "malos_capturados_top30%": t["%_malos_acum"].iloc[2],
        "ks_max": t["ks"].max(),
        "decil_ks_max": int(t["ks"].idxmax()),
    }


def psi(esperado: pd.Series, actual: pd.Series, n_tramos: int = 10) -> float:
    """Population Stability Index entre dos muestras de una variable.

    Los tramos salen de la muestra esperada (entrenamiento); los nulos son un tramo.
    Guía: < 0,1 estable · 0,1–0,25 cambio moderado · > 0,25 cambio fuerte.
    """
    eps = 1e-6
    if pd.api.types.is_numeric_dtype(esperado):
        _, cortes = pd.qcut(esperado.dropna(), n_tramos, retbins=True, duplicates="drop")
        cortes[0], cortes[-1] = -np.inf, np.inf
        a = pd.cut(esperado, cortes).astype("string").fillna("NULO")
        b = pd.cut(actual, cortes).astype("string").fillna("NULO")
    else:
        a = esperado.astype("string").fillna("NULO")
        b = actual.astype("string").fillna("NULO")
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True).reindex(pa.index.union(b.unique()), fill_value=0)
    pa = pa.reindex(pb.index, fill_value=0)
    pa, pb = pa + eps, pb + eps
    return float(((pb - pa) * np.log(pb / pa)).sum())

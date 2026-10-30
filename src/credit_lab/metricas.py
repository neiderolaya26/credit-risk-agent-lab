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

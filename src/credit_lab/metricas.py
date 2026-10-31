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


def bootstrap_auc(y, scores: dict, referencia: str, n: int = 300, semilla: int = 0) -> pd.DataFrame:
    """Diferencia de AUC de cada score contra `referencia`, con IC del 95% por bootstrap pareado.

    Pareado = en cada réplica se remuestrean los mismos préstamos para todos los
    modelos, así la comparación no se contamina por la suerte de la muestra.
    """
    y = np.asarray(y)
    scores = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    rng = np.random.default_rng(semilla)
    difs = {k: [] for k in scores if k != referencia}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        base = roc_auc_score(y[i], scores[referencia][i])
        for k in difs:
            difs[k].append(roc_auc_score(y[i], scores[k][i]) - base)
    filas = []
    for k, d in difs.items():
        d = np.array(d)
        filas.append({"modelo": k, "dif_auc": roc_auc_score(y, scores[k]) - roc_auc_score(y, scores[referencia]),
                      "ic95_inf": np.percentile(d, 2.5), "ic95_sup": np.percentile(d, 97.5),
                      "prob_no_mejora": float((d <= 0).mean())})
    return pd.DataFrame(filas).set_index("modelo")


def tabla_calibracion(y, pd_estimada, n: int = 10) -> pd.DataFrame:
    """PD promedio estimada vs. tasa de malos observada, por decil de PD."""
    y, p = pd.Series(np.asarray(y)), pd.Series(np.asarray(pd_estimada, dtype=float))
    decil = pd.qcut(p.rank(method="first"), n, labels=range(1, n + 1))
    t = pd.DataFrame({"pd_estimada": p.groupby(decil.values).mean(),
                      "tasa_real": y.groupby(decil.values).mean()})
    t["diferencia"] = t["tasa_real"] - t["pd_estimada"]
    t.index.name = "decil_pd"
    return t


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

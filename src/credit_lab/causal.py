"""Efecto causal de la tasa en la mora, el prepago y el rendimiento (decisión 014).

Experimento natural: Lending Club fijaba la tasa por subgrado con tablas que
cambiaba en meses distintos para cada subgrado. Dentro de una celda
subgrado × mes casi todos pagan la misma tasa; entre meses, la tasa del mismo
subgrado sube o baja por decisión de Lending Club, no por el cliente.

- Efectos fijos de dos vías (subgrado + mes): compara el mismo subgrado antes y
  después de un cambio de tabla, contra los subgrados que no cambiaron ese mes
  (diff-in-diff con tratamiento continuo). Supuesto: sin el cambio de tabla, la
  mora de todos los subgrados se habría movido en paralelo.
- DML (double machine learning): Y = θ·T + g(X) + ε y T = m(X) + v. LightGBM
  predice Y y T con el perfil del cliente, el subgrado y el mes; θ sale de
  regresar el residuo de Y en el residuo de T (cross-fitting en 5 pliegues).
  El mes entra solo de forma aditiva (interaction_constraints): si pudiera
  interactuar con el subgrado, el modelo aprendería la tabla de tasas completa
  y no quedaría variación para identificar θ.
- Placebo: la tasa del mes siguiente no puede causar la mora de hoy.

Resultado (decisión 015): el placebo falla con efectos fijos y con DML. Lending
Club cambiaba sus tablas reaccionando a cómo se movía cada subgrado, así que la
variación no es aleatoria y el efecto en la mora no queda identificado. El motor
mantiene la PD fija ante la tasa y agrega un escenario de estrés y una bandera
para tasas fuera del rango observado.

Lo que se estima es el efecto total de cobrar más: la cuota más pesada y la
selección adversa (quién acepta un precio más alto) juntas. Para el motor
ambos importan igual. No hay datos de aceptación, así que no se estima la
demanda.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from credit_lab.features import ARBOLES, derivar
from credit_lab.prepago import CORTE_PREPAGO, meses_pago, rendimiento
from credit_lab.transformadores import preprocesador

RESULTADOS = ["malo", "prepago", "rendimiento"]
SUBGRADOS = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
INICIO_MES = 2012 * 12 + 9
SEMILLA = 0
EFECTO_ESTRES = 0.01   # +1 pt de PD por punto de sobreprecio: mayor estimación puntual (notebook 10)
RANGO_OBSERVADO = 1.0  # puntos: la tasa residual de las tablas se movió ±0,74 (p1–p99)
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=500,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEMILLA)


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega mes (0 = sep-2012), prepago (pagó todo antes del mes 30), rendimiento y celda."""
    df = df.copy()
    df["mes"] = df["issue_d"].dt.year * 12 + df["issue_d"].dt.month - INICIO_MES
    buenos = df["malo"].eq(0)
    df["prepago"] = 0
    df.loc[buenos, "prepago"] = (meses_pago(df[buenos]) < CORTE_PREPAGO).astype(int)
    df["rendimiento"] = rendimiento(df)
    df["celda"] = df["sub_grade"].astype(str) + "_" + df["mes"].astype(str)
    return df


# ---------- regresión con errores agrupados ----------

def quitar_efectos_fijos(datos: pd.DataFrame, grupos: list, tol: float = 1e-8, max_iter: int = 200) -> pd.DataFrame:
    """Resta las medias de cada efecto fijo por turnos hasta converger (proyecciones alternadas).
    Equivale a meter una dummy por subgrado y por mes, sin construir la matriz de dummies."""
    x = datos.astype(float).copy()
    for _ in range(max_iter):
        antes = x.to_numpy().copy()
        for g in grupos:
            x = x - x.groupby(g).transform("mean")
        if np.abs(x.to_numpy() - antes).max() < tol:
            break
    return x


def ols_cluster(y, X, cluster, k_absorbidos: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """MCO con varianza robusta agrupada (CR1). `k_absorbidos` = efectos fijos quitados antes."""
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    n, k = X.shape
    pan = np.linalg.inv(X.T @ X)
    b = pan @ X.T @ y
    e = y - X @ b
    s = pd.DataFrame(X * e[:, None]).groupby(np.asarray(cluster)).sum().to_numpy()
    g = len(s)
    c = g / (g - 1) * (n - 1) / (n - k - k_absorbidos)
    v = c * pan @ (s.T @ s) @ pan
    return b, np.sqrt(np.diag(v))


def efectos_fijos(df: pd.DataFrame, y: str, controles: pd.DataFrame | None = None,
                  fijos: tuple = ("sub_grade", "mes"), cluster: str = "sub_grade",
                  tratamientos: tuple = ("int_rate",)) -> pd.DataFrame:
    """Coeficiente de cada tratamiento (en unidades de y por punto de tasa) con IC 95% agrupado por subgrado."""
    datos = pd.DataFrame({"y": df[y].to_numpy()} | {t: df[t].to_numpy() for t in tratamientos}, index=df.index)
    if controles is not None:
        datos = datos.join(controles.set_axis(df.index))
    grupos = [df[f].to_numpy() for f in fijos]
    if grupos:
        x = quitar_efectos_fijos(datos, grupos)
        k_abs = sum(len(np.unique(g)) for g in grupos) - (len(grupos) - 1)
    else:
        x = datos - datos.mean()
        k_abs = 1
    b, se = ols_cluster(x["y"], x.drop(columns="y"), df[cluster], k_abs)
    k = len(tratamientos)
    return pd.DataFrame({"theta": b[:k], "se": se[:k], "ic_bajo": b[:k] - 1.96 * se[:k],
                         "ic_alto": b[:k] + 1.96 * se[:k]}, index=list(tratamientos))


def tasa_siguiente(df: pd.DataFrame) -> pd.Series:
    """Placebo: tasa media del mismo subgrado el mes siguiente. Un cambio de tabla futuro no
    puede causar la mora de hoy; si su coeficiente no es ≈ 0, las tendencias no eran paralelas."""
    media = df.groupby(["sub_grade", "mes"])["int_rate"].mean()
    sig = media.rename(index=lambda m: m - 1, level="mes")
    return pd.Series(pd.MultiIndex.from_frame(df[["sub_grade", "mes"]]).map(sig), index=df.index)


# ---------- DML ----------

class Controles:
    """Perfil del cliente (variables de los árboles) + subgrado ordinal + mes."""

    def fit(self, df):
        self.pre = preprocesador("arboles").fit(derivar(df)[ARBOLES], df["malo"])
        return self

    def transform(self, df):
        X = self.pre.transform(derivar(df)[ARBOLES])
        X["subgrado"] = df["sub_grade"].map({s: i for i, s in enumerate(SUBGRADOS)}).to_numpy()
        X["mes"] = df["mes"].to_numpy()
        return X


def _restricciones(columnas) -> list:
    """El mes no interactúa con nada: su efecto es común a todos (como un efecto fijo de mes)."""
    i_mes = list(columnas).index("mes")
    return [[i for i in range(len(columnas)) if i != i_mes], [i_mes]]


def _modelo(binario: bool, columnas):
    clase = lgb.LGBMClassifier if binario else lgb.LGBMRegressor
    return clase(**PARAMS, interaction_constraints=_restricciones(columnas))


def residuos(X: pd.DataFrame, t, ys: dict, pliegues: int = 5) -> pd.DataFrame:
    """Residuos fuera de pliegue de la tasa (v) y de cada resultado (u_<nombre>)."""
    t = np.asarray(t, dtype=float)
    out = {"v": np.empty(len(X))} | {f"u_{k}": np.empty(len(X)) for k in ys}
    for fit, pred in KFold(pliegues, shuffle=True, random_state=SEMILLA).split(X):
        m = _modelo(False, X.columns).fit(X.iloc[fit], t[fit])
        out["v"][pred] = t[pred] - m.predict(X.iloc[pred])
        for k, y in ys.items():
            y = np.asarray(y, dtype=float)
            binario = set(np.unique(y)) <= {0.0, 1.0}
            m = _modelo(binario, X.columns).fit(X.iloc[fit], y[fit])
            yhat = m.predict_proba(X.iloc[pred])[:, 1] if binario else m.predict(X.iloc[pred])
            out[f"u_{k}"][pred] = y[pred] - yhat
    return pd.DataFrame(out, index=X.index)


def theta(u, v, cluster) -> dict:
    """θ = Σ u·v / Σ v², con varianza agrupada (función de influencia)."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    th = (u * v).sum() / (v * v).sum()
    psi = pd.Series(v * (u - th * v)).groupby(np.asarray(cluster)).sum().to_numpy()
    se = np.sqrt((psi ** 2).sum()) / (v * v).sum()
    return {"theta": th, "se": se, "ic_bajo": th - 1.96 * se, "ic_alto": th + 1.96 * se}


# ---------- integración al motor (decisión 015) ----------

def pd_estres(pd_base, tasa, tasa_ref, efecto: float = EFECTO_ESTRES) -> np.ndarray:
    """Escenario de estrés: la PD sube `efecto` por cada punto que la tasa supera la de mercado.
    Solo castiga el sobreprecio; cobrar menos no baja la PD (conservador)."""
    exceso = np.maximum(np.asarray(tasa, dtype=float) - np.asarray(tasa_ref, dtype=float), 0)
    return np.clip(np.asarray(pd_base, dtype=float) + efecto * exceso, 0, 0.999)


def fuera_de_rango(tasa, tasa_ref, rango: float = RANGO_OBSERVADO) -> np.ndarray:
    """La tasa se aleja de la de mercado más de lo que se movieron las tablas: ahí no hay evidencia."""
    return np.abs(np.asarray(tasa, dtype=float) - np.asarray(tasa_ref, dtype=float)) > rango

"""Prepago: el "churn" del crédito (decisión 010). Tres formulaciones que
terminan en lo mismo: el rendimiento de un préstamo **si sale bueno** (r_bueno).

- Línea base: promedio de r_bueno por subgrado (lo que usa la política hoy).
- A · Clasificación: P(prepago antes del mes 30) →
      r_bueno = P × r_prepago(subgrado) + (1 − P) × r_completo(subgrado)
- B · Regresión: predice r_bueno directamente.
- C · Supervivencia en tiempo discreto: P(pagar todo en el mes t | sigue vivo)
      para t = 1..35 → distribución del mes de pago → interés esperado con la
      tabla de amortización real del préstamo.

Población: solo préstamos buenos (el default ya lo cubre la PD).
Variables: las de los árboles + int_rate (aquí se modela comportamiento dado el
precio, no riesgo; ver decisión 010).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from credit_lab.features import ARBOLES, derivar
from credit_lab.negocio import utilidad_prestamo
from credit_lab.transformadores import preprocesador

PLAZO = 36
CORTE_PREPAGO = 30
SEMILLA = 0
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=SEMILLA)


# ---------- objetivos ----------

def meses_pago(df: pd.DataFrame) -> pd.Series:
    """Mes en que se pagó todo (1..36). Pagos tardíos (> 36) cuentan como plazo completo."""
    fin = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y").dt.to_period("M")
    m = (fin - df["issue_d"].dt.to_period("M")).apply(lambda x: x.n)
    return m.clip(1, PLAZO).astype(int)


def rendimiento(df: pd.DataFrame) -> pd.Series:
    return utilidad_prestamo(df) / df["funded_amnt"]


def interes_hasta_mes(tasa_anual, mes) -> np.ndarray:
    """Interés acumulado (fracción del monto) de un préstamo a 36 meses que se paga todo en `mes`."""
    j = np.asarray(tasa_anual, dtype=float) / 100 / 12
    k = np.arange(1, PLAZO + 1)
    factor = (1 + j[:, None]) ** PLAZO
    saldo_previo = (factor - (1 + j[:, None]) ** (k - 1)) / (factor - 1)  # saldo al inicio del mes k
    interes_mes = j[:, None] * saldo_previo
    acumulado = np.cumsum(interes_mes, axis=1)
    mes = np.clip(np.asarray(mes, dtype=int), 1, PLAZO)
    return acumulado[np.arange(len(j)), mes - 1] if mes.ndim else acumulado[:, mes - 1]


# ---------- variables ----------

class Variables:
    """Pipeline de árboles + int_rate, ajustado con entrenamiento."""

    def fit(self, df):
        self.pre = preprocesador("arboles").fit(derivar(df)[ARBOLES], df["malo"])
        return self

    def transform(self, df):
        X = self.pre.transform(derivar(df)[ARBOLES])
        X["int_rate"] = df["int_rate"].to_numpy()
        return X


def _validacion(df):
    """Últimos 3 meses del entrenamiento como validación temporal para early stopping."""
    corte = df["issue_d"] >= "2014-10-01"
    return df[~corte], df[corte]


def _lgbm(objetivo, X, y, Xv, yv, **extra):
    m = (lgb.LGBMClassifier if objetivo == "binary" else lgb.LGBMRegressor)(
        n_estimators=3000, **PARAMS, **extra)
    m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


# ---------- línea base ----------

class LineaBase:
    def fit(self, buenos):
        self.r_ = rendimiento(buenos).groupby(buenos["sub_grade"]).mean()
        return self

    def predecir(self, df):
        return df["sub_grade"].map(self.r_).fillna(self.r_.mean()).to_numpy()


# ---------- A · clasificación ----------

class Clasificacion:
    def fit(self, buenos):
        self.var = Variables().fit(buenos)
        r, pp = rendimiento(buenos), meses_pago(buenos) < CORTE_PREPAGO
        g = pd.DataFrame({"sg": buenos["sub_grade"], "pp": pp, "r": r}).groupby(["sg", "pp"])["r"].mean().unstack()
        self.r_prepago_, self.r_completo_ = g[True], g[False]
        fit, val = _validacion(buenos)
        self.modelo = _lgbm("binary", self.var.transform(fit), (meses_pago(fit) < CORTE_PREPAGO).astype(int),
                            self.var.transform(val), (meses_pago(val) < CORTE_PREPAGO).astype(int))
        return self

    def prob_prepago(self, df):
        return self.modelo.predict_proba(self.var.transform(df))[:, 1]

    def predecir(self, df):
        p = self.prob_prepago(df)
        rp = df["sub_grade"].map(self.r_prepago_).fillna(self.r_prepago_.mean()).to_numpy()
        rc = df["sub_grade"].map(self.r_completo_).fillna(self.r_completo_.mean()).to_numpy()
        return p * rp + (1 - p) * rc


# ---------- B · regresión ----------

class Regresion:
    def fit(self, buenos):
        self.var = Variables().fit(buenos)
        fit, val = _validacion(buenos)
        self.modelo = _lgbm("regression", self.var.transform(fit), rendimiento(fit),
                            self.var.transform(val), rendimiento(val), objective="l1")
        return self

    def predecir(self, df):
        return self.modelo.predict(self.var.transform(df))


# ---------- C · supervivencia en tiempo discreto ----------

def _persona_mes(X: pd.DataFrame, meses: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Expande cada préstamo en una fila por mes vivo (1..min(mes, 35)). Evento = pagó todo ese mes."""
    hasta = np.minimum(meses, PLAZO - 1)
    idx = np.repeat(np.arange(len(X)), hasta)
    t = np.concatenate([np.arange(1, h + 1) for h in hasta])
    filas = X.iloc[idx].reset_index(drop=True)
    filas["mes"] = t
    evento = (t == meses[idx]).astype(int)  # pagar en el mes 36 es el vencimiento normal, no un evento
    return filas, evento


class Supervivencia:
    def __init__(self, max_prestamos: int = 80_000):
        self.max_prestamos = max_prestamos

    def fit(self, buenos):
        self.var = Variables().fit(buenos)
        fit, val = _validacion(buenos)
        fit = fit.sample(min(self.max_prestamos, len(fit)), random_state=SEMILLA)
        Xf, yf = _persona_mes(self.var.transform(fit), meses_pago(fit).to_numpy())
        Xv, yv = _persona_mes(self.var.transform(val), meses_pago(val).to_numpy())
        self.modelo = _lgbm("binary", Xf, yf, Xv, yv)
        return self

    def distribucion(self, df) -> np.ndarray:
        """Probabilidad de pagar todo en cada mes 1..36 (filas = préstamos)."""
        X = self.var.transform(df)
        n = len(X)
        riesgo = np.empty((n, PLAZO - 1))
        for t in range(1, PLAZO):
            Xt = X.copy()
            Xt["mes"] = t
            riesgo[:, t - 1] = self.modelo.predict_proba(Xt)[:, 1]
        vivo = np.cumprod(np.c_[np.ones(n), 1 - riesgo], axis=1)  # P(seguir vivo al inicio del mes t)
        p = np.c_[riesgo * vivo[:, :-1], vivo[:, -1]]  # el resto paga en el mes 36
        return p

    def predecir(self, df):
        p = self.distribucion(df)
        tasa = df["int_rate"].to_numpy()
        interes = np.column_stack([interes_hasta_mes(tasa, np.full(len(df), m)) for m in range(1, PLAZO + 1)])
        return (p * interes).sum(axis=1)

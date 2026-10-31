"""Severidad: cuánto se pierde si el préstamo cae (decisión 010). Dos formulaciones
que terminan en lo mismo: el rendimiento de un préstamo **si sale malo** (r_malo).

Un préstamo malo, por cada unidad prestada:
    r_malo = (capital devuelto + intereses + recuperado − comisión − monto) / monto
Con EAD = monto − capital devuelto y LGD = (EAD − recuperado + comisión) / EAD:
    r_malo = intereses/monto − (EAD/monto) × LGD

- Línea base: promedio de r_malo por subgrado (lo que usa la política hoy).
- A · LGD clásica (Basilea): modela la LGD; intereses/monto y EAD/monto se toman
  del promedio por subgrado → r_malo = interés_medio − EAD_media × LGD predicha.
- B · Pérdida económica: predice r_malo directamente.

Población: solo préstamos malos.
"""

import numpy as np
import pandas as pd

from credit_lab.prepago import Variables, _lgbm, _validacion, rendimiento


def ead(df):
    return df["funded_amnt"] - df["total_rec_prncp"]


def lgd(df) -> pd.Series:
    """Si ya devolvió todo el capital (EAD = 0) no hay exposición que perder: LGD = 0."""
    e = ead(df)
    l = (e - df["recoveries"] + df["collection_recovery_fee"]) / e.where(e > 0)
    return l.fillna(0).clip(0, 1)


class LineaBase:
    def fit(self, malos):
        self.r_ = rendimiento(malos).groupby(malos["sub_grade"]).mean()
        return self

    def predecir(self, df):
        return df["sub_grade"].map(self.r_).fillna(self.r_.mean()).to_numpy()


class LGDClasica:
    def fit(self, malos):
        self.var = Variables().fit(malos)
        g = malos.groupby("sub_grade")
        self.interes_ = (malos["total_rec_int"] / malos["funded_amnt"]).groupby(malos["sub_grade"]).mean()
        self.ead_ = (ead(malos) / malos["funded_amnt"]).groupby(malos["sub_grade"]).mean()
        fit, val = _validacion(malos)
        self.modelo = _lgbm("regression", self.var.transform(fit), lgd(fit),
                            self.var.transform(val), lgd(val), objective="l1")
        return self

    def predecir_lgd(self, df):
        return np.clip(self.modelo.predict(self.var.transform(df)), 0, 1)

    def predecir(self, df):
        i = df["sub_grade"].map(self.interes_).fillna(self.interes_.mean()).to_numpy()
        e = df["sub_grade"].map(self.ead_).fillna(self.ead_.mean()).to_numpy()
        return i - e * self.predecir_lgd(df)


class PerdidaEconomica:
    def fit(self, malos):
        self.var = Variables().fit(malos)
        fit, val = _validacion(malos)
        self.modelo = _lgbm("regression", self.var.transform(fit), rendimiento(fit),
                            self.var.transform(val), rendimiento(val), objective="l1")
        return self

    def predecir(self, df):
        return self.modelo.predict(self.var.transform(df))

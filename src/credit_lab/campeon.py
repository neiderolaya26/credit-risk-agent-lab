"""Modelo campeón (decisión 006: LightGBM) con recalibración (decisión 007).

La recalibración mueve el intercepto en escala logit para que la PD media
iguale la tasa de malos observada en el periodo más reciente disponible
(1er semestre de 2015). El orden de los préstamos no cambia, así que el AUC
tampoco: solo se corrige el nivel de la PD.

Uso:
    python -m credit_lab.campeon      # calcula el desplazamiento y guarda modelos/campeon.joblib
"""

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from credit_lab.features import derivar
from credit_lab.poblacion import RAIZ, cargar_poblacion
from credit_lab.retadores import predecir

NOMBRE = "lgbm"
RUTA = RAIZ / "modelos" / "campeon.joblib"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibrar(pd_modelo, desplazamiento: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(_logit(pd_modelo) + desplazamiento)))


def desplazamiento_intercepto(pd_modelo, y) -> float:
    """Desplazamiento en logit tal que la PD media recalibrada = tasa real."""
    z, objetivo = _logit(pd_modelo), float(np.mean(y))
    return brentq(lambda d: np.mean(1 / (1 + np.exp(-(z + d)))) - objetivo, -5, 5)


class Campeon:
    def __init__(self, pre, modelo, desplazamiento: float):
        self.pre, self.modelo, self.desplazamiento = pre, modelo, desplazamiento

    def pd_sin_calibrar(self, df: pd.DataFrame) -> np.ndarray:
        return predecir(NOMBRE, self.pre, self.modelo, derivar(df))

    def pd(self, df: pd.DataFrame) -> np.ndarray:
        return recalibrar(self.pd_sin_calibrar(df), self.desplazamiento)


def cargar() -> Campeon:
    # Se guardan los componentes y no la clase, para que el archivo no dependa
    # de cómo se ejecutó el script que lo creó.
    return Campeon(**joblib.load(RUTA))


def main() -> None:
    pre, modelo = joblib.load(RAIZ / "modelos" / f"{NOMBRE}.joblib")
    df = cargar_poblacion()
    s1 = df[(df["muestra"] == "oot") & (df["issue_d"].dt.month <= 6)]
    campeon = Campeon(pre, modelo, 0.0)
    d = desplazamiento_intercepto(campeon.pd_sin_calibrar(s1), s1["malo"])
    joblib.dump({"pre": pre, "modelo": modelo, "desplazamiento": d}, RUTA)
    print(f"Campeón guardado en {RUTA} · desplazamiento logit = {d:.4f}")


if __name__ == "__main__":
    main()

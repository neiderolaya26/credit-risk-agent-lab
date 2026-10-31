"""Motor de cupo y pricing (decisión 012).

Para cada cliente se simula una grilla de montos y tasas:
    PD(monto)            → campeón recalibrado (el monto entra por monto_ingreso)
    r_bueno(tasa)        → supervivencia: con más tasa, más prepago
    r_malo(tasa)         → pérdida económica
    utilidad esperada    = monto × [(1 − PD) × r_bueno + PD × r_malo]
    capacidad de pago    = cuota mensual / ingreso mensual ≤ 20%

Salidas (decisión 013):
    monto máximo     → mayor monto con utilidad > 0 y capacidad de pago, a una tasa dada
    tasa equilibrio  → menor tasa con ROI esperado ≥ 0%: por debajo pierde dinero (piso de rechazo)
    tasa sugerida    → menor tasa con ROI esperado ≥ 4%: cubre fondeo y operación, que la
                       utilidad bruta no descuenta (precio sugerido)

Supuestos: la PD no reacciona a la tasa (decisión 001) y no hay datos de
aceptación del cliente (precio sin elasticidad).

Decisión 015: el efecto causal de la tasa en la mora no se pudo identificar
(notebook 10). Por eso:
    tasa_mercado     → la que cobraría Lending Club a ese perfil (TasaMercado recalibrada)
    fuera_de_rango   → la tasa se aleja más de 1 pt de la de mercado: sin evidencia
    estrés           → opcional: la PD sube 1 pt por punto de sobreprecio
"""

import joblib
import numpy as np
import pandas as pd

from credit_lab.campeon import cargar as cargar_campeon
from credit_lab.causal import fuera_de_rango, pd_estres
from credit_lab.poblacion import RAIZ

PLAZO = 36
CUOTA_MAX = 0.20
ROI_EQUILIBRIO = 0.0
ROI_OBJETIVO = 0.04
MONTOS = np.arange(1000, 35001, 1000)
TASAS = np.round(np.arange(5.0, 30.01, 0.5), 2)


def cuota(monto, tasa_anual, plazo: int = PLAZO):
    j = np.asarray(tasa_anual, dtype=float) / 100 / 12
    return np.asarray(monto, dtype=float) * j / (1 - (1 + j) ** -plazo)


class Motor:
    def __init__(self, cuota_max: float = CUOTA_MAX, estres: bool = False):
        m = RAIZ / "modelos"
        self.campeon = cargar_campeon()
        self.supervivencia = joblib.load(m / "prepago_c_supervivencia.joblib")
        self.perdida = joblib.load(m / "severidad_b_economica.joblib")
        self.tasa_mercado = joblib.load(m / "referencia_tasa.joblib")
        self.cuota_max = cuota_max
        self.estres = estres

    def evaluar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evalúa cada fila tal cual viene (su loan_amnt e int_rate)."""
        df = df.copy()
        df["funded_amnt"] = df["loan_amnt"]
        tasa, ref = df["int_rate"].to_numpy(), self.tasa_mercado.predecir_recalibrada(df)
        p = self.campeon.pd(df)
        if self.estres:
            p = pd_estres(p, tasa, ref)
        rb = self.supervivencia.predecir(df)
        rm = self.perdida.predecir(df)
        out = pd.DataFrame({"monto": df["loan_amnt"].to_numpy(), "tasa": tasa, "tasa_mercado": ref,
                            "fuera_de_rango": fuera_de_rango(tasa, ref),
                            "pd": p, "r_bueno": rb, "r_malo": rm}, index=df.index)
        out["roi_esperado"] = (1 - p) * rb + p * rm
        out["utilidad_esperada"] = out["monto"] * out["roi_esperado"]
        out["cuota_ingreso"] = cuota(out["monto"], out["tasa"]) / (df["annual_inc"].to_numpy() / 12)
        out["cumple_capacidad"] = out["cuota_ingreso"] <= self.cuota_max
        return out

    @staticmethod
    def _fila(cliente) -> pd.DataFrame:
        """Una fila como DataFrame: repetir un Series pierde los tipos de las columnas."""
        return cliente if isinstance(cliente, pd.DataFrame) else cliente.to_frame().T.infer_objects()

    def _grilla(self, cliente: pd.DataFrame, montos, tasas) -> pd.DataFrame:
        g = cliente.loc[cliente.index.repeat(len(montos) * len(tasas))].reset_index(drop=True)
        mm, tt = np.meshgrid(montos, tasas, indexing="ij")
        g["loan_amnt"], g["int_rate"] = mm.ravel().astype(float), tt.ravel()
        return g

    def simular(self, cliente, montos=MONTOS, tasas=TASAS) -> pd.DataFrame:
        """Utilidad esperada de todas las combinaciones monto × tasa para un cliente (fila de 1)."""
        return self.evaluar(self._grilla(self._fila(cliente), montos, tasas))

    def monto_maximo(self, sim: pd.DataFrame, tasa: float) -> float:
        ok = sim[np.isclose(sim["tasa"], tasa) & (sim["utilidad_esperada"] > 0) & sim["cumple_capacidad"]]
        return float(ok["monto"].max()) if len(ok) else 0.0

    def tasa_minima(self, sim: pd.DataFrame, monto: float, roi_objetivo: float = 0.0) -> float:
        ok = sim[np.isclose(sim["monto"], monto) & (sim["roi_esperado"] >= roi_objetivo)]
        return float(ok["tasa"].min()) if len(ok) else np.nan

    def recomendar(self, cliente, roi_objetivo: float = ROI_OBJETIVO) -> dict:
        """Monto máximo a la tasa ofrecida, tasa de equilibrio y tasa sugerida para el monto pedido."""
        cliente = self._fila(cliente)
        pedido, tasa_ofrecida = float(cliente["loan_amnt"].iloc[0]), float(cliente["int_rate"].iloc[0])
        montos = np.unique(np.r_[MONTOS, pedido])
        tasas = np.unique(np.r_[TASAS, tasa_ofrecida])
        sim = self.simular(cliente, montos, tasas)
        actual = sim[np.isclose(sim["monto"], pedido) & np.isclose(sim["tasa"], tasa_ofrecida)].iloc[0]
        return {
            "monto_pedido": pedido,
            "tasa_ofrecida": tasa_ofrecida,
            "tasa_mercado": float(actual["tasa_mercado"]),
            "pd": float(actual["pd"]),
            "roi_esperado": float(actual["roi_esperado"]),
            "utilidad_esperada": float(actual["utilidad_esperada"]),
            "cuota_ingreso": float(actual["cuota_ingreso"]),
            "monto_maximo": self.monto_maximo(sim, tasa_ofrecida),
            "tasa_equilibrio": self.tasa_minima(sim, pedido, ROI_EQUILIBRIO),
            "tasa_sugerida": (sugerida := self.tasa_minima(sim, pedido, roi_objetivo)),
            "sugerida_fuera_de_rango": bool(np.isfinite(sugerida)
                                            and fuera_de_rango(sugerida, actual["tasa_mercado"])),
        }


# ---------- versión por lotes (backtest) ----------

def _expandir(df: pd.DataFrame, columna: str, valores) -> pd.DataFrame:
    g = df.loc[df.index.repeat(len(valores))].copy()
    g[columna] = np.tile(np.asarray(valores, dtype=float), len(df))
    g["_prestamo"] = np.repeat(np.arange(len(df)), len(valores))
    return g


def tasa_minima_lote(motor: Motor, df: pd.DataFrame, roi_objetivo: float = 0.0, tasas=TASAS) -> np.ndarray:
    """Para cada préstamo, con su monto pedido: menor tasa con ROI esperado ≥ objetivo."""
    g = _expandir(df, "int_rate", tasas)
    ev = motor.evaluar(g.drop(columns="_prestamo"))
    ok = ev["roi_esperado"].to_numpy() >= roi_objetivo
    tasa = np.where(ok, g["int_rate"].to_numpy(), np.inf)
    minimo = pd.Series(tasa).groupby(g["_prestamo"].to_numpy()).min().to_numpy()
    return np.where(np.isinf(minimo), np.nan, minimo)


def monto_maximo_lote(motor: Motor, df: pd.DataFrame, montos=MONTOS) -> np.ndarray:
    """Para cada préstamo, a su tasa de Lending Club: mayor monto con utilidad > 0 y capacidad de pago."""
    g = _expandir(df, "loan_amnt", montos)
    ev = motor.evaluar(g.drop(columns="_prestamo"))
    ok = (ev["utilidad_esperada"].to_numpy() > 0) & ev["cumple_capacidad"].to_numpy()
    monto = np.where(ok, g["loan_amnt"].to_numpy(), 0.0)
    return pd.Series(monto).groupby(g["_prestamo"].to_numpy()).max().to_numpy()

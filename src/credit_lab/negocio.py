"""Métricas de negocio: cuánto dinero gana o pierde la cartera según el modelo.

Utilidad realizada de cada préstamo (flujos reales de Lending Club):
    total_pymnt - funded_amnt - collection_recovery_fee

`total_pymnt` ya incluye capital, intereses, cargos por mora y recuperaciones
post-castigo (verificado: coincide en el 100% de los préstamos de 2015).
No se descuenta el costo de fondeo ni el operativo, así que es una utilidad
bruta: sirve para comparar modelos entre sí, no como rentabilidad final.
"""

import numpy as np
import pandas as pd


def utilidad_prestamo(df: pd.DataFrame) -> pd.Series:
    return df["total_pymnt"] - df["funded_amnt"] - df["collection_recovery_fee"]


def utilidad_por_corte(utilidad, monto, score, tasas_aprobacion=None) -> pd.DataFrame:
    """Utilidad de aprobar solo el `tasa` % de menor riesgo según `score`.

    `score` debe crecer con el riesgo. Se aprueba de menor a mayor score.
    """
    if tasas_aprobacion is None:
        tasas_aprobacion = np.arange(0.5, 1.0001, 0.05)
    orden = np.argsort(np.asarray(score, dtype=float), kind="stable")
    u = np.asarray(utilidad, dtype=float)[orden]
    m = np.asarray(monto, dtype=float)[orden]
    filas = []
    for tasa in tasas_aprobacion:
        k = int(round(tasa * len(u)))
        filas.append({"tasa_aprobacion": tasa, "prestamos": k, "monto": m[:k].sum(),
                      "utilidad": u[:k].sum(), "roi": u[:k].sum() / m[:k].sum()})
    return pd.DataFrame(filas).set_index("tasa_aprobacion")

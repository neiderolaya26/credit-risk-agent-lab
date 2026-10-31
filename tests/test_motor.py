import numpy as np
import pandas as pd

from credit_lab.motor import _expandir, cuota


def test_cuota_de_amortizacion():
    assert np.isclose(cuota(10_000, 12), 332.14, atol=0.01)


def test_cuota_sube_con_monto_y_tasa():
    assert cuota(20_000, 12) > cuota(10_000, 12)
    assert cuota(10_000, 20) > cuota(10_000, 12)


def test_expandir_crea_una_fila_por_valor_y_conserva_tipos():
    df = pd.DataFrame({"loan_amnt": [5000.0, 8000.0], "emp_length": pd.Series(["2 years", None], dtype="string")})
    g = _expandir(df, "int_rate", [6.0, 9.0, 12.0])
    assert len(g) == 6
    assert g["_prestamo"].tolist() == [0, 0, 0, 1, 1, 1]
    assert g["int_rate"].tolist() == [6.0, 9.0, 12.0] * 2
    assert g["emp_length"].dtype == "string"


def test_monto_maximo_con_roi_minimo_es_mas_estricto():
    from credit_lab.motor import Motor

    sim = pd.DataFrame({"tasa": [12.0] * 3, "monto": [5_000.0, 10_000.0, 15_000.0],
                        "utilidad_esperada": [300.0, 250.0, 100.0], "roi_esperado": [0.06, 0.025, 0.007],
                        "cumple_capacidad": [True, True, True]})
    assert Motor.monto_maximo(None, sim, 12.0) == 15_000.0
    assert Motor.monto_maximo(None, sim, 12.0, roi_minimo=0.04) == 5_000.0

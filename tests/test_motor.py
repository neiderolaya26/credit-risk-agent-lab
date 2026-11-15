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


# ---------- la política que decide (decisión 023), con un motor simulado: sin modelos ni datos ----------

class _Fijo:
    """Modelos de mentira: PD 10%, pérdida −50% si sale malo, rendimiento bueno ≈ la tasa, y la tasa de
    mercado viene en la columna `ref`. Así ROI ≥ 4% equivale a tasa ≥ 10%."""

    def pd(self, df):
        return np.full(len(df), 0.10)

    def predecir(self, df):
        return df["int_rate"].to_numpy(float) / 100 + 0.001

    def predecir_recalibrada(self, df):
        return df["ref"].to_numpy(float)


class _Malo(_Fijo):
    def predecir(self, df):
        return np.full(len(df), -0.50)


def _motor_simulado():
    from credit_lab.motor import CUOTA_MAX, Motor

    m = Motor.__new__(Motor)  # sin __init__: no carga los .joblib
    m.campeon = m.supervivencia = m.tasa_mercado = _Fijo()
    m.perdida = _Malo()
    m.cuota_max, m.estres, m.efecto_estres = CUOTA_MAX, False, 0.0
    return m


def _solicitudes(**columnas):
    base = {"loan_amnt": 10_000.0, "int_rate": 12.0, "annual_inc": 60_000.0, "ref": 12.0}
    n = max(len(v) for v in columnas.values())
    return pd.DataFrame({k: columnas.get(k, [v] * n) for k, v in base.items()})


def test_politica_aplica_las_reglas_en_orden():
    from credit_lab.motor import politica_lote

    sol = _solicitudes(loan_amnt=[10_000.0, 35_000.0, 10_000.0, 10_000.0],
                       int_rate=[12.0, 12.0, 9.0, 7.0],
                       annual_inc=[60_000.0, 24_000.0, 60_000.0, 60_000.0],
                       ref=[12.0, 12.0, 9.5, 7.0])
    p = politica_lote(_motor_simulado(), sol)
    assert p["decision"].tolist() == ["aprobar", "contraoferta", "contraoferta", "rechazar"]
    assert p["regla"].tolist() == ["1", "2b", "2a", "3"]
    # 2b baja el monto hasta que la cuota cabe en el 20% del ingreso; 2a sube la tasa a la sugerida
    assert p["monto"].tolist() == [10_000.0, 12_000.0, 10_000.0, 0.0]
    assert p["tasa"].tolist() == [12.0, 12.0, 10.0, 0.0]


def test_politica_no_sube_la_tasa_mas_del_tope_ni_fuera_de_la_banda():
    from credit_lab.causal import RANGO_OBSERVADO
    from credit_lab.motor import TOPE_SUBIDA, politica_lote

    motor = _motor_simulado()
    lejos = _solicitudes(int_rate=[10.0 - TOPE_SUBIDA - 0.5], ref=[10.0])  # la sugerida (10%) supera el tope
    assert politica_lote(motor, lejos)["decision"].tolist() == ["rechazar"]
    fuera = _solicitudes(int_rate=[9.0], ref=[10.0 - RANGO_OBSERVADO - 1])  # la sugerida queda fuera de la banda
    assert politica_lote(motor, fuera)["decision"].tolist() == ["rechazar"]
    assert politica_lote(motor, fuera, banda=False)["regla"].tolist() == ["2a"]


def test_lo_que_la_politica_otorga_cumple_las_reglas_duras():
    from credit_lab.motor import CUOTA_MAX, ROI_EQUILIBRIO, decidir_politica

    motor = _motor_simulado()
    for monto, tasa, ingreso in [(10_000.0, 12.0, 60_000.0), (35_000.0, 12.0, 24_000.0), (10_000.0, 9.0, 60_000.0)]:
        d = decidir_politica(motor, _solicitudes(loan_amnt=[monto], int_rate=[tasa], annual_inc=[ingreso], ref=[tasa]))
        oferta = motor.evaluar(_solicitudes(loan_amnt=[d["monto"]], int_rate=[d["tasa"]], annual_inc=[ingreso], ref=[tasa])).iloc[0]
        assert oferta["cuota_ingreso"] <= CUOTA_MAX and oferta["roi_esperado"] >= ROI_EQUILIBRIO

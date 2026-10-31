import numpy as np
import pandas as pd

from credit_lab.metricas import bootstrap_auc, tabla_calibracion
from credit_lab.negocio import utilidad_por_corte, utilidad_prestamo


def test_utilidad_descuenta_monto_y_comision():
    df = pd.DataFrame({"total_pymnt": [1200.0, 400.0], "funded_amnt": [1000.0, 1000.0],
                       "collection_recovery_fee": [0.0, 20.0]})
    assert utilidad_prestamo(df).tolist() == [200.0, -620.0]


def test_aprobar_por_menor_riesgo_mejora_la_utilidad():
    utilidad = np.array([100, 100, 100, -500])
    score = np.array([0.1, 0.2, 0.3, 0.9])  # el último es el más riesgoso y el que pierde
    t = utilidad_por_corte(utilidad, np.ones(4) * 1000, score, [0.75, 1.0])
    assert t.loc[0.75, "utilidad"] == 300 and t.loc[1.0, "utilidad"] == -200


def test_bootstrap_detecta_un_modelo_mejor():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    bueno = y + rng.normal(scale=0.8, size=5000)
    malo = y + rng.normal(scale=3.0, size=5000)
    r = bootstrap_auc(y, {"bueno": bueno, "malo": malo}, referencia="malo", n=50)
    assert r.loc["bueno", "ic95_inf"] > 0


def test_calibracion_perfecta():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.5, 50_000)
    y = (rng.uniform(size=50_000) < p).astype(int)
    assert tabla_calibracion(y, p)["diferencia"].abs().max() < 0.02


def test_recalibracion_iguala_la_tasa_y_conserva_el_orden():
    from credit_lab.campeon import desplazamiento_intercepto, recalibrar
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.4, 20_000)
    y = (rng.uniform(size=20_000) < p * 1.2).astype(int)  # la realidad es 20% más riesgosa
    d = desplazamiento_intercepto(p, y)
    p2 = recalibrar(p, d)
    assert d > 0
    assert abs(p2.mean() - y.mean()) < 1e-6
    assert (np.argsort(p) == np.argsort(p2)).all()

import numpy as np
import pandas as pd

from credit_lab.iv import information_value, tabla_woe


def test_variable_sin_relacion_tiene_iv_cercano_a_cero():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=20_000))
    y = pd.Series(rng.integers(0, 2, size=20_000))
    assert information_value(x, y) < 0.01


def test_variable_que_separa_tiene_iv_alto():
    x = pd.Series(["A"] * 500 + ["B"] * 500)
    y = pd.Series([0] * 450 + [1] * 50 + [0] * 50 + [1] * 450)
    assert information_value(x, y) > 0.5


def test_tramos_numericos_conservan_su_orden():
    x = pd.Series([1.0, 5.0, 9.0, 30.0, 120.0, 900.0] * 10)
    y = pd.Series([0, 1] * 30)
    inicios = [float(t.split(",")[0].strip("(")) for t in tabla_woe(x, y, n=3).index]
    assert inicios == sorted(inicios)


def test_nulos_forman_su_propio_tramo():
    x = pd.Series([1.0, 2.0, None, None])
    y = pd.Series([0, 1, 0, 1])
    assert "NULO" in tabla_woe(x, y, n=2).index

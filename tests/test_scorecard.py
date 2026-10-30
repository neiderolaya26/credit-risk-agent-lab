import numpy as np

from credit_lab.scorecard import BASE, ODDS_BASE, PDO, _inicio_tramo, pd_a_puntos


def test_tramos_se_ordenan_numericamente():
    tramos = ["(120000.0, inf]", "(-inf, 32000.0]", "(32000.0, 40000.0]", "NULO"]
    assert sorted(tramos, key=_inicio_tramo) == [
        "(-inf, 32000.0]", "(32000.0, 40000.0]", "(120000.0, inf]", "NULO"]


def test_escala_base():
    pd_base = 1 / (1 + ODDS_BASE)  # odds 50:1
    assert np.isclose(pd_a_puntos(pd_base), BASE)


def test_pdo_duplica_las_odds():
    odds = 100  # el doble de la base
    assert np.isclose(pd_a_puntos(1 / (1 + odds)), BASE + PDO)


def test_mas_riesgo_menos_puntos():
    assert pd_a_puntos(0.30) < pd_a_puntos(0.05)

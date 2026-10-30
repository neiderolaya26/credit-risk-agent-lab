import numpy as np
import pandas as pd

from credit_lab.metricas import psi
from credit_lab.transformadores import CENTINELA, Limpieza, Winsorizador, WoE


def test_limpieza_quita_centinela_y_agrupa_vivienda():
    X = pd.DataFrame({"total_rev_hi_lim": [1000.0, CENTINELA],
                      "home_ownership": ["ANY", "RENT"]})
    out = Limpieza().transform(X)
    assert np.isnan(out.loc[1, "total_rev_hi_lim"])
    assert out.loc[0, "home_ownership"] == "OTHER"


def test_winsorizador_usa_limites_de_entrenamiento():
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    w = Winsorizador().fit(train)
    out = w.transform(pd.DataFrame({"x": [-1000.0, 1000.0]}))
    assert out["x"].between(0, 99).all()


def test_woe_maneja_nulos_y_categorias_nuevas():
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, None, None] * 50,
                      "cat": ["a", "b", "a", "b", "a", "b"] * 50})
    y = [0, 0, 1, 1, 0, 1] * 50
    woe = WoE(n_tramos=2).fit(X, y)
    nuevo = pd.DataFrame({"num": [None, 99.0], "cat": ["z", "a"]})
    out = woe.transform(nuevo)
    assert out.notna().all().all()
    assert out.loc[0, "cat"] == 0.0  # categoría no vista = riesgo promedio


def test_psi_cero_si_misma_distribucion_y_alto_si_cambia():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(size=10_000))
    assert psi(a, a) < 0.001
    assert psi(a, a + 1) > 0.25

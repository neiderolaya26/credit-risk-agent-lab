import numpy as np
import pandas as pd

from credit_lab.causal import efectos_fijos, fuera_de_rango, ols_cluster, pd_estres, tasa_siguiente, theta


def _panel(n=4000, efecto=0.5, semilla=0):
    """Subgrados con nivel propio, meses con shock común y tasa que cambia por subgrado × mes."""
    rng = np.random.default_rng(semilla)
    sg = rng.integers(0, 10, n)
    mes = rng.integers(0, 12, n)
    tabla = rng.normal(0, 1, (10, 12))
    tasa = 10 + sg + 0.3 * mes + tabla[sg, mes]
    y = 2 * sg - 0.5 * mes + efecto * tasa + rng.normal(0, 1, n)
    return pd.DataFrame({"sub_grade": sg, "mes": mes, "int_rate": tasa, "y": y})


def test_ols_cluster_coincide_con_minimos_cuadrados():
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(500), rng.normal(size=(500, 2))]
    y = X @ [1.0, 2.0, -1.0] + rng.normal(size=500)
    b, se = ols_cluster(y, X, np.arange(500) % 20)
    assert np.allclose(b, np.linalg.lstsq(X, y, rcond=None)[0])
    assert (se > 0).all()


def test_efectos_fijos_recupera_el_efecto_y_la_correlacion_simple_no():
    df = _panel()
    dos_vias = efectos_fijos(df, "y").loc["int_rate"]
    simple = efectos_fijos(df, "y", fijos=()).loc["int_rate"]
    assert dos_vias["ic_bajo"] < 0.5 < dos_vias["ic_alto"]
    assert simple["theta"] > 1  # absorbe el nivel del subgrado: sesgo por selección


def test_theta_dml_con_residuos_conocidos():
    rng = np.random.default_rng(2)
    v = rng.normal(size=5000)
    u = 0.3 * v + rng.normal(0, 0.1, 5000)
    r = theta(u, v, np.arange(5000) % 50)
    assert abs(r["theta"] - 0.3) < 0.01 and r["ic_bajo"] < 0.3 < r["ic_alto"]


def test_tasa_siguiente_es_la_media_del_mismo_subgrado_el_mes_siguiente():
    df = pd.DataFrame({"sub_grade": ["A1", "A1", "A1", "B1"], "mes": [0, 1, 1, 0], "int_rate": [6.0, 7.0, 8.0, 9.0]})
    sig = tasa_siguiente(df)
    assert sig.iloc[0] == 7.5 and np.isnan(sig.iloc[1]) and np.isnan(sig.iloc[3])


def test_estres_solo_castiga_el_sobreprecio():
    p = pd_estres([0.10, 0.10], tasa=[15.0, 10.0], tasa_ref=[12.0, 12.0], efecto=0.01)
    assert np.allclose(p, [0.13, 0.10])


def test_fuera_de_rango():
    assert fuera_de_rango([12.0, 13.5, 10.5], [12.0, 12.0, 12.0], rango=1.0).tolist() == [False, True, True]

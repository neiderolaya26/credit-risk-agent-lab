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


def test_dml_de_punta_a_punta_recupera_theta_con_confusion(monkeypatch):
    """Issue #45: panel sintético donde el perfil del cliente mueve la tasa y la mora a la vez.
    El DML (residuos con cross-fitting + theta) recupera el efecto; la correlación simple no."""
    from credit_lab import causal

    monkeypatch.setattr(causal, "PARAMS", causal.PARAMS | {"n_estimators": 300, "learning_rate": 0.1, "min_child_samples": 20})
    rng = np.random.default_rng(3)
    n, efecto = 6000, 0.4
    sg, mes = rng.integers(0, 10, n), rng.integers(0, 12, n)
    x = rng.normal(size=n)  # perfil del cliente: confusor
    tasa = 10 + sg + 0.3 * mes + 1.5 * x + rng.normal(0, 1, n)
    y = 2 * sg - 0.5 * mes + 3 * x + efecto * tasa + rng.normal(0, 1, n)
    X = pd.DataFrame({"x": x, "subgrado": sg, "mes": mes})
    r = causal.residuos(X, tasa, {"y": y}, pliegues=3)
    dml = theta(r["u_y"], r["v"], sg * 12 + mes)
    simple = np.polyfit(tasa, y, 1)[0]
    assert dml["ic_bajo"] < efecto < dml["ic_alto"]
    assert abs(simple - efecto) > 0.5  # la correlación simple absorbe el confusor


def test_las_restricciones_dejan_al_mes_sin_interacciones():
    from credit_lab.causal import _restricciones

    assert _restricciones(["x", "subgrado", "mes"]) == [[0, 1], [2]]


def test_wild_cluster_bootstrap_cubre_el_efecto_y_rechaza_cero():
    from credit_lab.causal import wild_cluster_bootstrap

    df = _panel(n=3000, efecto=0.5, semilla=4)
    X = np.c_[np.ones(len(df)), df["int_rate"]]
    w = wild_cluster_bootstrap(df["y"] - 2 * df["sub_grade"] + 0.5 * df["mes"], X, df["sub_grade"], j=1, replicas=199)
    assert w["ic_bajo"] < 0.5 < w["ic_alto"] and w["p_valor"] < 0.05


def test_dos_vias_no_es_mas_angosta_que_una():
    from credit_lab.causal import ols_cluster_dos_vias

    df = _panel(n=3000, semilla=5)
    X = np.c_[np.ones(len(df)), df["int_rate"]]
    _, se1 = ols_cluster(df["y"], X, df["sub_grade"])
    _, se2 = ols_cluster_dos_vias(df["y"], X, df["sub_grade"], df["mes"])
    assert (se2 >= se1 - 1e-12).all()

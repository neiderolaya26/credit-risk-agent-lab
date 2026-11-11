import numpy as np

from credit_lab.valor import aprobados, comparar, por_grupo


def test_aprobados_rechaza_la_fraccion_mas_riesgosa():
    assert aprobados(np.arange(100), 0.2).sum() == 80


def test_comparar_premia_al_score_que_ordena_mejor():
    rng = np.random.default_rng(0)
    malo = rng.random(20_000) < 0.15
    utilidad = np.where(malo, -3_000.0, 800.0)
    bueno_score = malo + rng.normal(0, 0.3, 20_000)  # ordena bien
    mal_score = rng.normal(0, 1, 20_000)             # ruido
    t = comparar(utilidad, malo, bueno_score, mal_score, rechazos=[0.2], replicas=100)
    assert t.loc[0.2, "diferencia"] > 0 and t.loc[0.2, "ic_bajo"] > 0
    assert t.loc[0.2, "mora_campeon"] < t.loc[0.2, "mora_lending_club"]


def test_por_grupo_mide_rechazo_y_calibracion():
    pd_modelo = np.r_[0.1 + np.linspace(-0.01, 0.01, 2000), 0.3 + np.linspace(-0.01, 0.01, 2000)]  # sin empates
    malo = np.r_[np.zeros(1800), np.ones(200), np.zeros(1400), np.ones(600)]
    grupo = np.r_[["a"] * 2000, ["b"] * 2000]
    t = por_grupo(pd_modelo, malo, grupo, tasa_rechazo=0.25)
    assert t.loc["b", "rechazo"] > t.loc["a", "rechazo"]
    assert abs(t.loc["a", "brecha_calibracion"]) < 1e-9 and abs(t.loc["b", "brecha_calibracion"]) < 1e-9
    assert t.loc["a", "rechazo"] == 0 and t.loc["b", "rechazo"] == 0.5


def _backtest_sintetico():
    import pandas as pd

    return pd.DataFrame({
        "loan_amnt": [10_000.0, 10_000.0, 10_000.0, 10_000.0], "int_rate": [10.0, 10.0, 10.0, 10.0],
        "utilidad_real": [1_000.0, 1_000.0, 1_000.0, -5_000.0], "roi_real": [0.1, 0.1, 0.1, -0.5],
        "total_rec_int": [1_500.0, 1_500.0, 1_500.0, 500.0], "malo": [0, 0, 0, 1],
        "pol_regla": ["1", "2b", "2a", "3"], "pol_monto": [10_000.0, 5_000.0, 10_000.0, 0.0],
        "pol_tasa": [10.0, 10.0, 12.0, 0.0], "sub_grade": ["A1", "B1", "C1", "G5"]})


def test_utilidad_de_la_politica_segun_la_aceptacion():
    from credit_lab.valor import _utilidad_politica

    b = _backtest_sintetico()
    u0, m0, k0 = _utilidad_politica(b, 0.0)
    assert list(k0) == [1, 0, 0, 0] and u0.sum() == 1_000.0
    u1, m1, _ = _utilidad_politica(b, 1.0)
    # 2b: 10% sobre 5.000; 2a: 1.000 + 1.500 × 2/10 de intereses − 0,01 × 2 × 0,6 × 10.000 de castigo
    assert u1[1] == 500.0 and abs(u1[2] - (1_000 + 300 - 120)) < 1e-9 and u1[3] == 0
    assert list(m1) == [10_000.0, 5_000.0, 10_000.0, 0.0]


def test_escenarios_y_equidad_en_precio():
    from credit_lab.valor import COSTOS, equidad_precio, escenarios

    b = _backtest_sintetico()
    e = escenarios(b, COSTOS | {"fondeo_anual": 0.0, "capital": 0.0, "operativo_prestamo": 0}, replicas=50)
    assert e.loc["Lending Club aprueba todo", "utilidad neta"] == -2_000 * 10_000 / 4
    assert e.loc["Política real · acepta 0% de contraofertas", "utilidad neta"] == 1_000 * 10_000 / 4
    t = equidad_precio(b.assign(g=["x", "x", "y", "y"]), "g", minimo=1)
    assert t.loc["y", "sube_tasa"] == 0.5 and t.loc["x", "rechazo"] == 0

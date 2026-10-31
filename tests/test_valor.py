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

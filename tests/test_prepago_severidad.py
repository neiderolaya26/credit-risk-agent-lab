import numpy as np
import pandas as pd

from credit_lab.prepago import PLAZO, _persona_mes, interes_hasta_mes
from credit_lab.severidad import lgd


def test_interes_total_coincide_con_la_cuota():
    cuota = 0.01 / (1 - 1.01 ** -36)  # 12% anual
    assert np.isclose(interes_hasta_mes([12.0], 36)[0], cuota * 36 - 1)


def test_prepagar_antes_cobra_menos_interes():
    i = interes_hasta_mes([15.0, 15.0], np.array([6, 30]))
    assert i[0] < i[1]


def test_persona_mes_marca_el_evento_y_no_el_vencimiento():
    X = pd.DataFrame({"x": [1.0, 2.0]})
    filas, evento = _persona_mes(X, np.array([3, PLAZO]))
    assert len(filas) == 3 + (PLAZO - 1)
    assert evento[:3].tolist() == [0, 0, 1]
    assert evento[3:].sum() == 0  # pagar en el mes 36 no es prepago


def test_lgd_clasica():
    df = pd.DataFrame({"funded_amnt": [1000.0], "total_rec_prncp": [400.0],
                       "recoveries": [60.0], "collection_recovery_fee": [10.0]})
    assert np.isclose(lgd(df).iloc[0], (600 - 60 + 10) / 600)


def test_lgd_sin_exposicion_es_cero():
    df = pd.DataFrame({"funded_amnt": [1000.0], "total_rec_prncp": [1000.0],
                       "recoveries": [0.0], "collection_recovery_fee": [0.0]})
    assert lgd(df).iloc[0] == 0

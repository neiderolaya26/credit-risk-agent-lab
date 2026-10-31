import numpy as np
import pandas as pd

from credit_lab.politica import aprobar, rendimientos_por_subgrado, utilidad_esperada


def _train():
    # Subgrado A: buenos ganan 10%, malos pierden 40%. Subgrado G: buenos ganan 40%, malos pierden 40%.
    return pd.DataFrame({
        "sub_grade": ["A1", "A1", "G1", "G1"], "malo": [0, 1, 0, 1],
        "funded_amnt": [1000.0] * 4, "total_pymnt": [1100.0, 600.0, 1400.0, 600.0],
        "collection_recovery_fee": [0.0] * 4,
    })


def test_pd_de_equilibrio_sube_con_el_precio():
    r = rendimientos_por_subgrado(_train())
    assert np.isclose(r.loc["A1", "pd_equilibrio"], 0.2)
    assert np.isclose(r.loc["G1", "pd_equilibrio"], 0.5)


def test_misma_pd_se_aprueba_en_subgrado_caro_y_no_en_barato():
    r = rendimientos_por_subgrado(_train())
    nuevos = pd.DataFrame({"sub_grade": ["A1", "G1"], "funded_amnt": [1000.0, 1000.0]})
    pd30 = [0.30, 0.30]
    assert aprobar(nuevos, pd30, r).tolist() == [False, True]
    assert np.allclose(utilidad_esperada(nuevos, pd30, r), [1000 * (0.7 * 0.1 - 0.3 * 0.4), 1000 * (0.7 * 0.4 - 0.3 * 0.4)])


def test_rendimientos_publicados_cubren_todos_los_subgrados():
    from credit_lab.politica import cargar_rendimientos
    r = cargar_rendimientos()
    assert len(r) == 35
    assert (r["r_bueno"] > 0).all() and (r["r_malo"] < 0).all()

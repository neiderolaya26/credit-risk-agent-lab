import pandas as pd

from credit_lab.features import ARBOLES, EXCLUIDAS, LOGISTICA, SIGNO_INCORRECTO, derivar
from credit_lab.poblacion import BENCHMARK, FUGA


def test_logistica_es_subconjunto_de_arboles():
    assert set(LOGISTICA) <= set(ARBOLES)


def test_signo_incorrecto_sale_de_logistica_pero_no_de_arboles():
    assert not set(SIGNO_INCORRECTO) & set(LOGISTICA)
    assert set(SIGNO_INCORRECTO) <= set(ARBOLES)


def test_sin_fuga_ni_benchmark_ni_excluidas():
    prohibidas = set(FUGA) | set(BENCHMARK) | set(EXCLUIDAS)
    assert not prohibidas & set(ARBOLES)


def test_derivar_calcula_variables_nuevas():
    df = pd.DataFrame({
        "fico_range_low": [700.0], "fico_range_high": [704.0],
        "issue_d": [pd.Timestamp("2014-01-01")], "earliest_cr_line": ["Jan-2004"],
        "loan_amnt": [10_000.0], "annual_inc": [50_000.0],
    })
    out = derivar(df)
    assert out.loc[0, "fico_medio"] == 702
    assert out.loc[0, "monto_ingreso"] == 0.2
    assert 119 < out.loc[0, "antig_credito_meses"] < 121

import pandas as pd

from credit_lab.evals import fidelidad, numeros_memo
from credit_lab.evals_runner import tabla_md


def test_numeros_en_formato_espanol_e_ingles():
    assert numeros_memo("Aprobar USD 24.000 a 5,32 % (PD 7,65 %); cuota 8,67 % de 20 %") == [24000.0, 5.32, 7.65, 8.67, 20.0]
    assert numeros_memo("USD 24,000 at 13.33% · ROI 0.044 · 1.234,5") == [24000.0, 13.33, 0.044, 1234.5]


def test_fidelidad_acepta_fracciones_como_porcentaje_y_detecta_inventados():
    salidas = [{"pd": 0.0765, "monto_maximo": 35000.0}, {"roi_esperado": 0.044362}]
    f, sin = fidelidad("PD 7,65 %, ROI 4,44 %, cupo USD 35.000 y una tasa inventada de 13,9 %", salidas, {})
    assert sin == [13.9] and abs(f - 3 / 4) < 1e-9


def test_fidelidad_ignora_conteos_pequenos_y_constantes_de_politica():
    f, sin = fidelidad("Las 5 razones bajan el riesgo; la cuota cumple el límite de 20 %.", [], {})
    assert f == 1.0 and sin == []


def test_tabla_md():
    t = tabla_md(pd.DataFrame({"a": [0.5]}, index=pd.Index(["x"], name="k")), "{:.1f}")
    assert t.splitlines() == ["| k | a |", "|---|---|", "| x | 0.5 |"]


def test_tabla_md_formatea_por_columna():
    t = tabla_md(pd.DataFrame({"préstamos": [24.0], "mora real": [0.375]}, index=["Agente"]),
                 {"préstamos": "{:,.0f}", "mora real": "{:.1%}"})
    assert t.splitlines()[-1] == "| Agente | 24 | 37.5% |"

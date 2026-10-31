import asyncio

import pandas as pd
import pytest

from credit_lab import servidor_mcp
from credit_lab.servidor_mcp import _Contexto, servidor


class HerramientasFalsas:
    def riesgo(self, sol):
        return {"pd": 0.1, "monto": float(sol["loan_amnt"].iloc[0])}

    def economia(self, sol):
        return {"roi_esperado": 0.05}

    def cupo_pricing(self, sol):
        return {"monto_maximo": 20_000.0}

    def simular_oferta(self, sol, monto, tasa):
        return {"monto": monto, "tasa": tasa}


@pytest.fixture(autouse=True)
def contexto_falso(monkeypatch):
    muestra = pd.DataFrame({"loan_amnt": [5_000.0, 9_000.0], "int_rate": [10.0, 14.0], "annual_inc": [50_000.0, 70_000.0]})
    monkeypatch.setattr(_Contexto, "herramientas", HerramientasFalsas())
    monkeypatch.setattr(_Contexto, "muestra", muestra)


def test_expone_las_herramientas_del_motor():
    nombres = {t.name for t in asyncio.run(servidor.list_tools())}
    assert nombres == {"buscar_solicitud", "consultar_riesgo", "consultar_economia",
                       "consultar_cupo_pricing", "simular_oferta"}


def test_las_herramientas_usan_el_prestamo_del_indice():
    assert servidor_mcp.consultar_riesgo(1)["monto"] == 9_000.0
    assert servidor_mcp.simular_oferta(0, 4_000, 11.5) == {"monto": 4_000, "tasa": 11.5}
    assert servidor_mcp.buscar_solicitud(0)["indice"] in (0, 1)


def test_indice_fuera_de_rango():
    with pytest.raises(ValueError):
        servidor_mcp.consultar_economia(5)


def test_llamada_por_protocolo():
    r = asyncio.run(servidor.call_tool("consultar_cupo_pricing", {"indice": 0}))
    assert not r.is_error

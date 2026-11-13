import asyncio

import pandas as pd
import pytest

from mcp.server.mcpserver.exceptions import ToolError

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

    def politica(self, sol):
        return {"decision": "aprobar", "regla": "1"}


@pytest.fixture(autouse=True)
def contexto_falso(monkeypatch):
    muestra = pd.DataFrame({"loan_amnt": [5_000.0, 9_000.0], "int_rate": [10.0, 14.0], "annual_inc": [50_000.0, 70_000.0]})
    monkeypatch.setattr(_Contexto, "herramientas", HerramientasFalsas())
    monkeypatch.setattr(_Contexto, "muestra", muestra)


def test_expone_las_herramientas_del_motor():
    nombres = {t.name for t in asyncio.run(servidor.list_tools())}
    assert nombres == {"buscar_solicitud", "consultar_riesgo", "consultar_economia",
                       "consultar_cupo_pricing", "simular_oferta", "decidir"}


def test_las_herramientas_son_de_solo_lectura():
    """Issue #33."""
    assert all(t.annotations.read_only_hint for t in asyncio.run(servidor.list_tools()))


def test_las_herramientas_usan_el_prestamo_del_indice():
    assert servidor_mcp.consultar_riesgo(1)["monto"] == 9_000.0
    assert servidor_mcp.simular_oferta(0, 4_000, 11.5) == {"monto": 4_000, "tasa": 11.5}
    assert servidor_mcp.buscar_solicitud(0)["indice"] in (0, 1)


def test_indice_fuera_de_rango():
    with pytest.raises(ToolError):
        servidor_mcp.consultar_economia(5)


def test_simular_oferta_fuera_de_la_grilla_devuelve_error_legible():
    """Issue #27: por protocolo, el cliente recibe un error y no una extrapolación."""
    with pytest.raises(ToolError):
        servidor_mcp.simular_oferta(0, -5, 12)
    with pytest.raises(ToolError, match="monto fuera de rango"):  # el protocolo lo devuelve como is_error
        asyncio.run(servidor.call_tool("simular_oferta", {"indice": 0, "monto": 1e9, "tasa": 12}))


def test_llamada_por_protocolo():
    r = asyncio.run(servidor.call_tool("consultar_cupo_pricing", {"indice": 0}))
    assert not r.is_error

"""Servidor MCP del motor de crédito (decisión 017).

Expone las mismas herramientas que usan los especialistas del agente (sesión 11)
para que cualquier cliente MCP (Claude Desktop, Claude Code, otro agente) las
use. Son código puro sobre los modelos: no llaman a ningún LLM ni necesitan
API key.

Los préstamos salen de la muestra pública de jul–dic 2015 (fuera de tiempo) y
se identifican por su posición en esa muestra, nunca por un id de Lending Club.

Uso:
    python -m credit_lab.servidor_mcp      # transporte stdio
"""

from typing import Any

from mcp.server.mcpserver import MCPServer

from credit_lab.agente import Herramientas, resumen_solicitud
from credit_lab.motor import CUOTA_MAX, ROI_EQUILIBRIO, ROI_OBJETIVO

POLITICA = f"""Política del motor de crédito (credit-risk-agent-lab):
- PD: LightGBM recalibrado (decisiones 006-007); no usa la tasa ni el score de Lending Club.
- ROI esperado = (1 − PD) × r_bueno + PD × r_malo (prepago por supervivencia y pérdida económica, decisión 011).
- Capacidad de pago: cuota ≤ {CUOTA_MAX:.0%} del ingreso (decisión 012).
- Tasa de equilibrio: ROI ≥ {ROI_EQUILIBRIO:.0%}; tasa sugerida: ROI ≥ {ROI_OBJETIVO:.0%} (decisión 013).
- El efecto causal de la tasa en la mora no se pudo identificar: la PD es fija ante la tasa, con un
  escenario de estrés (+1 pt de PD por punto de sobreprecio) y una bandera fuera de rango
  (> 1 pt de la tasa de mercado) (decisión 015)."""

servidor = MCPServer(
    name="credit-lab",
    title="Motor de crédito Lending Club",
    instructions="Herramientas de riesgo, economía, cupo y pricing sobre préstamos públicos de Lending Club "
                 "(jul–dic 2015). Primero usa buscar_solicitud para obtener un índice.",
)


class _Contexto:
    """Carga perezosa: los modelos y los datos se leen en la primera llamada."""

    herramientas = None
    muestra = None

    @classmethod
    def listo(cls):
        if cls.muestra is None:
            from credit_lab.poblacion import cargar_poblacion

            df = cargar_poblacion()
            cls.muestra = df[df["issue_d"] >= "2015-07-01"].reset_index(drop=True)
        if cls.herramientas is None:
            cls.herramientas = Herramientas()
        return cls.herramientas, cls.muestra


def _solicitud(indice: int):
    herr, muestra = _Contexto.listo()
    if not 0 <= indice < len(muestra):
        raise ValueError(f"índice fuera de rango: use un valor entre 0 y {len(muestra) - 1}")
    return herr, muestra.iloc[[indice]]


@servidor.tool(structured_output=True)
def buscar_solicitud(semilla: int = 0) -> dict[str, Any]:
    """Elige un préstamo al azar de jul–dic 2015 y devuelve su índice y el resumen de la solicitud."""
    _, muestra = _Contexto.listo()
    indice = int(muestra.sample(1, random_state=semilla).index[0])
    return {"indice": indice, "solicitud": resumen_solicitud(muestra.iloc[[indice]])}


@servidor.tool(structured_output=True)
def consultar_riesgo(indice: int) -> dict[str, Any]:
    """PD del modelo campeón y las 5 razones SHAP que más la mueven."""
    herr, sol = _solicitud(indice)
    return herr.riesgo(sol)


@servidor.tool(structured_output=True)
def consultar_economia(indice: int) -> dict[str, Any]:
    """Rendimiento si sale bueno (prepago) y si sale malo (pérdida), y ROI esperado a la tasa ofrecida."""
    herr, sol = _solicitud(indice)
    return herr.economia(sol)


@servidor.tool(structured_output=True)
def consultar_cupo_pricing(indice: int) -> dict[str, Any]:
    """Monto máximo, tasas de equilibrio y sugerida (base y con estrés) y capacidad de pago."""
    herr, sol = _solicitud(indice)
    return herr.cupo_pricing(sol)


@servidor.tool(structured_output=True)
def simular_oferta(indice: int, monto: float, tasa: float) -> dict[str, Any]:
    """PD, ROI esperado, cuota/ingreso y bandera fuera de rango para otro monto (USD) y tasa (%)."""
    herr, sol = _solicitud(indice)
    return herr.simular_oferta(sol, monto, tasa)


@servidor.resource("credito://politica", name="politica", mime_type="text/plain",
                   description="Reglas y supuestos del motor de crédito.")
def politica() -> str:
    return POLITICA


def main() -> None:
    servidor.run("stdio")


if __name__ == "__main__":
    main()

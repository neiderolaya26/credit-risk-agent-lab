"""Agente de crédito 100% IA en LangGraph (decisión 016).

    solicitud ─┬─ especialista de riesgo      (Claude + PD y razones SHAP)
               ├─ especialista de economía    (Claude + prepago y pérdida)
               └─ especialista de cupo/pricing (Claude + motor, estrés y simulación)
                      ↓ (en paralelo)
               comité (Claude: decide y redacta el memo)
                      ↓
               reglas duras (código: cuota ≤ 20% del ingreso y ROI esperado ≥ 0%)
                      ↓
               auditor (Claude: verifica números y reglas) ── observaciones → comité
                      ↓
               decisión final

Los números salen siempre de las herramientas (código sobre los modelos del
repo); los LLM interpretan, deciden y redactan. Si el auditor no aprueba tras
INTENTOS rondas, la decisión final es rechazar (conservador).

Registro y observabilidad (decisión 018): cada ejecución se guarda en SQLite
(credit_lab.registro) con el resultado real del préstamo, y si existe
LANGSMITH_API_KEY el grafo y cada llamada a Claude se trazan en LangSmith.

Uso:
    python -m credit_lab.agente               # un préstamo de jul–dic 2015
    python -m credit_lab.agente --semilla 7   # otro préstamo
Requiere ANTHROPIC_API_KEY (variable de entorno o archivo .env, que no se sube a git).
"""

import argparse
import copy
import json
import math
import operator
import os
import threading
import time
from typing import Annotated, Literal, TypedDict

import numpy as np
import pandas as pd
from pydantic import BaseModel

from credit_lab.causal import RANGO_OBSERVADO
from credit_lab.motor import CUOTA_MAX, ROI_EQUILIBRIO, ROI_OBJETIVO, Motor, cuota
from credit_lab.poblacion import RAIZ

MODELO = "claude-sonnet-5-5"
VERSION = "v3"  # v0: piloto inicial; v1: política compartida, FICO del modelo, criterios explícitos, auditor por materialidad; v2: memo sin invenciones; v3: regla 2b con ROI ≥ 4% (decisión 020)
# USD por millón de tokens (entrada, salida)
PRECIOS = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}
BETAS = ["server-side-fallback-2026-07-01"]
CON_FALLBACK = {"claude-sonnet-5-5", "claude-opus-5-5"}  # Haiku 5.5 no tiene fallback del servidor
INTENTOS = 2
MAX_TURNOS = 6
MONTO_MAX = 35_000
TASA_MIN, TASA_MAX = 5.0, 30.0

CAMPOS_SOLICITUD = {
    "loan_amnt": "monto pedido (USD)", "int_rate": "tasa ofrecida por Lending Club (%)",
    "annual_inc": "ingreso anual (USD)", "purpose": "propósito", "dti": "deuda / ingreso (%)",
    "emp_length": "antigüedad laboral", "home_ownership": "vivienda",
    "revol_util": "uso del crédito rotativo (%)", "inq_last_6mths": "consultas últimos 6 meses",
}
CAMPO_FICO = "FICO (punto medio del rango, el que usa el modelo)"
CRITERIOS = {
    "criterio_fuera_de_rango": f"la tasa se aleja más de {RANGO_OBSERVADO:g} pt de la tasa de mercado; ahí no hay evidencia histórica",
    "nota_tasa_mercado": "la tasa de mercado se estima para el perfil y el monto, así que cambia si cambia el monto",
}


# ---------- utilidades ----------

def cargar_env(ruta=RAIZ / ".env") -> None:
    """Lee KEY=VALOR de .env sin sobrescribir variables ya definidas. Nunca imprime valores."""
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if "=" in linea and not linea.lstrip().startswith("#"):
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def limpio(x):
    """Convierte tipos de numpy/pandas a JSON (NaN → None)."""
    if isinstance(x, dict):
        return {k: limpio(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpio(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is pd.NA or x is pd.NaT:
        return None
    return x


def resumen_solicitud(fila: pd.DataFrame) -> dict:
    f = fila.iloc[0]
    r = {nombre: limpio(f.get(campo)) for campo, nombre in CAMPOS_SOLICITUD.items()}
    bajo, alto = f.get("fico_range_low"), f.get("fico_range_high")
    r[CAMPO_FICO] = limpio((bajo + alto) / 2 if alto is not None and pd.notna(alto) else bajo)
    return r


# ---------- herramientas: código puro sobre los modelos ----------

class Herramientas:
    def __init__(self, motor: Motor | None = None):
        from credit_lab.explicar import explicar_prestamo

        self.motor = motor or Motor()
        self.motor_estres = copy.copy(self.motor)
        self.motor_estres.estres = True
        self._explicar = explicar_prestamo

    def riesgo(self, sol: pd.DataFrame) -> dict:
        r = self._explicar(self.motor.campeon, sol)
        return limpio({"pd": r["pd"], "pd_promedio_cartera": r["pd_base"],
                       "razones": [{k: x[k] for k in ("descripcion", "valor", "efecto", "contribucion_logit")}
                                   for x in r["razones"]]})

    def economia(self, sol: pd.DataFrame) -> dict:
        ev = self.motor.evaluar(sol).iloc[0]
        p = self.motor.supervivencia.distribucion(sol)[0]
        return limpio({"r_bueno": ev["r_bueno"], "r_malo": ev["r_malo"], "pd": ev["pd"],
                       "roi_esperado": ev["roi_esperado"], "utilidad_esperada": ev["utilidad_esperada"],
                       "prob_prepago_antes_mes_30": p[:29].sum(),
                       "mes_esperado_de_pago": float((np.arange(1, 37) * p).sum())})

    def cupo_pricing(self, sol: pd.DataFrame) -> dict:
        base, estres = self.motor.recomendar(sol), self.motor_estres.recomendar(sol)
        # las dos contraofertas de la política ya simuladas, para que se puedan verificar (v3)
        contraofertas = {}
        if pd.notna(base["tasa_sugerida"]):
            contraofertas["contraoferta_a_tasa_sugerida"] = self.simular_oferta(sol, base["monto_pedido"], base["tasa_sugerida"])
        if 0 < base["monto_maximo_objetivo"] < base["monto_pedido"]:
            contraofertas["contraoferta_b_monto_objetivo"] = self.simular_oferta(sol, base["monto_maximo_objetivo"], base["tasa_ofrecida"])
        return limpio(base | contraofertas | {"tasa_equilibrio_estres": estres["tasa_equilibrio"],
                              "tasa_sugerida_estres": estres["tasa_sugerida"],
                              "regla_cuota_max": CUOTA_MAX} | CRITERIOS)

    def simular_oferta(self, sol: pd.DataFrame, monto: float, tasa: float) -> dict:
        s = sol.copy()
        s["loan_amnt"], s["int_rate"] = float(monto), float(tasa)
        ev = self.motor.evaluar(s).iloc[0]
        return limpio({"monto": monto, "tasa": tasa, "pd": ev["pd"], "roi_esperado": ev["roi_esperado"],
                       "utilidad_esperada": ev["utilidad_esperada"], "cuota_ingreso": ev["cuota_ingreso"],
                       "cumple_capacidad": ev["cumple_capacidad"], "tasa_mercado": ev["tasa_mercado"],
                       "fuera_de_rango": ev["fuera_de_rango"]} | CRITERIOS)


SIN_ARGUMENTOS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
ESQUEMA_SIMULAR = {
    "type": "object",
    "properties": {"monto": {"type": "number", "description": "Monto en USD (1.000 a 35.000)"},
                   "tasa": {"type": "number", "description": "Tasa anual en % (5 a 30)"}},
    "required": ["monto", "tasa"], "additionalProperties": False,
}
SIMULAR = ("simular_oferta", "Calcula PD, ROI esperado, utilidad, cuota/ingreso y bandera fuera de rango "
           "para un monto y una tasa dados.", ESQUEMA_SIMULAR)

ESPECIALISTAS = {
    "riesgo": {
        "sistema": "Eres el especialista de riesgo de crédito. Usa la herramienta para obtener la PD del "
                   "modelo campeón (LightGBM recalibrado) y sus razones SHAP. Explica qué impulsa el riesgo.",
        "herramientas": [("consultar_riesgo", "PD del cliente y las 5 razones que más la mueven.", SIN_ARGUMENTOS)],
    },
    "economia": {
        "sistema": "Eres el especialista de economía del préstamo. Usa las herramientas para obtener el "
                   "rendimiento si sale bueno (prepago por supervivencia), el rendimiento si sale malo "
                   "(pérdida económica) y el ROI esperado. Explica qué tan rentable es a la tasa ofrecida.",
        "herramientas": [("consultar_economia", "Rendimientos, prepago esperado y ROI a la tasa ofrecida.", SIN_ARGUMENTOS),
                         SIMULAR],
    },
    "cupo": {
        "sistema": "Eres el especialista de cupo y pricing. Usa las herramientas para obtener el monto máximo "
                   "(utilidad > 0) y el monto máximo objetivo (ROI ≥ 4%), "
                   "la tasa de equilibrio (ROI ≥ 0%), la tasa sugerida (ROI ≥ 4%), el escenario de estrés "
                   "(la PD sube 1 pt por punto de sobreprecio) y la bandera fuera de rango (> 1 pt de la tasa "
                   "de mercado: sin evidencia). Puedes simular ofertas alternativas. Propón la mejor oferta.",
        "herramientas": [("consultar_cupo_pricing", "Monto máximo, tasas de equilibrio y sugerida, estrés y "
                          "capacidad de pago.", SIN_ARGUMENTOS), SIMULAR],
    },
}
INSTRUCCION_ESPECIALISTA = (
    "Responde en español, en máximo 150 palabras: hallazgos clave y una señal final "
    "(favorable / neutral / desfavorable). Cita solo números que devolvieron tus herramientas; "
    "no inventes datos."
)


# ---------- salidas estructuradas ----------

class Decision(BaseModel):
    decision: Literal["aprobar", "rechazar", "contraoferta"]
    monto: float
    tasa: float
    razones: list[str]
    memo: str


class Auditoria(BaseModel):
    aprobado: bool
    hallazgos: list[str]


POLITICA = f"""Política del banco (en este orden):
1. Aprobar si, al monto y la tasa ofrecidos, el ROI esperado es ≥ {ROI_OBJETIVO:.0%} y la cuota es ≤ {CUOTA_MAX:.0%} del ingreso.
2. Si no, contraoferta con la primera opción que cumpla:
   a. el monto pedido a la tasa sugerida (ROI ≥ {ROI_OBJETIVO:.0%}), si existe, está entre {TASA_MIN:g}% y {TASA_MAX:g}% y la cuota sigue ≤ {CUOTA_MAX:.0%};
   b. el monto máximo objetivo (el mayor monto con ROI ≥ {ROI_OBJETIVO:.0%} a la tasa ofrecida) si es menor que el pedido.
3. Si no, rechazar (monto = 0, tasa = 0).
Reglas duras que verifica el código: cuota ≤ {CUOTA_MAX:.0%} del ingreso y ROI esperado ≥ {ROI_EQUILIBRIO:.0%} en lo que se otorga.
El escenario de estrés y la bandera fuera de rango son alertas que se mencionan; por sí solas no cambian la decisión
(una tasa sugerida fuera de rango sigue siendo una contraoferta válida). Las contraofertas a y b vienen ya simuladas
en la herramienta de cupo.
El FICO de la solicitud es el punto medio del rango, el mismo que usa el modelo."""

SISTEMA_COMITE = f"""Eres el comité de crédito. Decides con base en el análisis de tres especialistas y los
resultados de sus herramientas, aplicando esta política:

{POLITICA}

El memo va en español, en markdown, con secciones: Resumen, Decisión, Riesgo, Economía, Cupo y precio, Alertas.
Cita solo números que aparecen en los resultados de las herramientas o en la política.
No agregues recomendaciones operativas (seguimiento, monitoreo, revisión manual) ni explicaciones causales que no
salgan de las herramientas o de la política. Atribuye a cada especialista solo lo que dijo. Antes de afirmar que un
dato no existe o no fue simulado, revisa los resultados de las herramientas."""

SISTEMA_AUDITOR = f"""Eres el auditor de crédito: reemplazas la revisión humana. La política vigente es:

{POLITICA}

Verifica que:
1. Cada número del memo coincide (con redondeo) con los resultados de las herramientas o con la política.
2. La decisión es la que exige la política para estos números.
3. No hay datos inventados ni contradicciones entre memo, razones y decisión.
4. Si hay violaciones de reglas duras, no puedes aprobar.
Rechaza solo por problemas materiales: un número incorrecto, una decisión distinta de la que exige la política,
una violación de reglas o un dato inventado que sostenga la decisión. Las observaciones menores (redacción,
énfasis, detalles que no cambian la decisión) van en hallazgos con el prefijo "Menor:" y no impiden aprobar."""


# ---------- reglas duras ----------

def reglas_duras(decision: dict, oferta: dict | None) -> list[str]:
    """Reglas que ningún LLM puede saltarse. Devuelve la lista de violaciones."""
    if decision["decision"] == "rechazar":
        return []
    v = []
    if not (0 < decision["monto"] <= MONTO_MAX):
        v.append(f"monto fuera de rango: {decision['monto']}")
    if not (TASA_MIN <= decision["tasa"] <= TASA_MAX):
        v.append(f"tasa fuera de la grilla: {decision['tasa']}")
    if oferta is not None:
        if oferta["cuota_ingreso"] is None or oferta["cuota_ingreso"] > CUOTA_MAX:
            v.append(f"cuota/ingreso {oferta['cuota_ingreso']} supera el máximo de {CUOTA_MAX:.0%}")
        if oferta["roi_esperado"] is None or oferta["roi_esperado"] < ROI_EQUILIBRIO:
            v.append(f"ROI esperado {oferta['roi_esperado']} menor que {ROI_EQUILIBRIO:.0%}")
    return v


# ---------- llamadas a Claude ----------

class Medidor:
    """Envuelve el cliente y suma tokens y llamadas de cada respuesta (seguro con nodos en paralelo)."""

    def __init__(self, cliente, modelo: str = MODELO):
        self._cliente, self._lock, self.modelo = cliente, threading.Lock(), modelo
        self.llamadas = self.tokens_entrada = self.tokens_salida = 0
        self.beta = type("Beta", (), {})()
        self.beta.messages = type("Mensajes", (), {})()
        self.beta.messages.create = self._medir(cliente.beta.messages.create)
        self.beta.messages.parse = self._medir(cliente.beta.messages.parse)

    def _medir(self, metodo):
        def envuelto(**kw):
            r = metodo(**kw)
            uso = getattr(r, "usage", None)
            with self._lock:
                self.llamadas += 1
                self.tokens_entrada += getattr(uso, "input_tokens", 0) or 0
                self.tokens_salida += getattr(uso, "output_tokens", 0) or 0
            return r
        return envuelto

    def reiniciar(self):
        with self._lock:
            self.llamadas = self.tokens_entrada = self.tokens_salida = 0

    def costo_usd(self) -> float:
        entrada, salida = PRECIOS[self.modelo]
        return (self.tokens_entrada * entrada + self.tokens_salida * salida) / 1e6


def _llamar(cliente, modelo: str = MODELO, **kw):
    """Todas las llamadas usan el modelo elegido y, si lo admite, la red de seguridad de fallback del servidor."""
    extra = {"betas": BETAS, "fallbacks": "default"} if modelo in CON_FALLBACK else {}
    return kw.pop("_metodo")(model=modelo, **extra, **kw)


def _texto(respuesta) -> str:
    return "".join(b.text for b in respuesta.content if b.type == "text").strip()


def correr_especialista(cliente, herr: Herramientas, nombre: str, sol: pd.DataFrame, modelo: str = MODELO) -> dict:
    cfg = ESPECIALISTAS[nombre]
    funciones = {
        "consultar_riesgo": lambda _: herr.riesgo(sol),
        "consultar_economia": lambda _: herr.economia(sol),
        "consultar_cupo_pricing": lambda _: herr.cupo_pricing(sol),
        "simular_oferta": lambda a: herr.simular_oferta(sol, a["monto"], a["tasa"]),
    }
    tools = [{"name": n, "description": d, "input_schema": e, "strict": True} for n, d, e in cfg["herramientas"]]
    mensajes = [{"role": "user", "content": "Solicitud:\n" + json.dumps(resumen_solicitud(sol), ensure_ascii=False)
                 + "\n\n" + INSTRUCCION_ESPECIALISTA}]
    llamadas = []
    for _ in range(MAX_TURNOS):
        r = _llamar(cliente, modelo, _metodo=cliente.beta.messages.create, max_tokens=8000,
                    system=cfg["sistema"], tools=tools, messages=mensajes,
                    output_config={"effort": "medium"})
        if r.stop_reason == "refusal":
            return {"analisis": "El modelo declinó analizar esta solicitud.", "herramientas": llamadas}
        mensajes.append({"role": "assistant", "content": r.content})
        usos = [b for b in r.content if b.type == "tool_use"]
        if not usos:
            break
        resultados = []
        for u in usos:
            try:
                salida = funciones[u.name](u.input)
                resultados.append({"type": "tool_result", "tool_use_id": u.id,
                                   "content": json.dumps(salida, ensure_ascii=False)})
                llamadas.append({"herramienta": u.name, "entrada": u.input, "salida": salida})
            except Exception as e:  # el error vuelve al modelo en vez de tumbar el flujo
                resultados.append({"type": "tool_result", "tool_use_id": u.id,
                                   "content": f"Error: {e}", "is_error": True})
        mensajes.append({"role": "user", "content": resultados})
    return {"analisis": _texto(r), "herramientas": llamadas}


def _contexto(estado) -> str:
    return json.dumps({"solicitud": resumen_solicitud(estado["solicitud"]),
                       "especialistas": estado["especialistas"]}, ensure_ascii=False, indent=1)


def decidir(cliente, estado, modelo: str = MODELO) -> dict:
    partes = [_contexto(estado)]
    if estado.get("auditoria") and not estado["auditoria"]["aprobado"]:
        partes.append("El auditor rechazó tu decisión anterior. Corrige estos hallazgos:\n- "
                      + "\n- ".join(estado["auditoria"]["hallazgos"] + estado.get("violaciones", [])))
    r = _llamar(cliente, modelo, _metodo=cliente.beta.messages.parse, max_tokens=16000, system=SISTEMA_COMITE,
                messages=[{"role": "user", "content": "\n\n".join(partes)}], output_format=Decision,
                output_config={"effort": "high"})
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return {"decision": "rechazar", "monto": 0.0, "tasa": 0.0,
                "razones": ["El comité no pudo emitir una decisión."], "memo": ""}
    return r.parsed_output.model_dump()


def auditar(cliente, estado, modelo: str = MODELO) -> dict:
    contenido = (_contexto(estado) + "\n\nDecisión del comité:\n"
                 + json.dumps(estado["decision"], ensure_ascii=False, indent=1)
                 + "\n\nOferta recalculada con el motor:\n" + json.dumps(estado.get("oferta"), ensure_ascii=False)
                 + "\n\nViolaciones de reglas duras:\n" + json.dumps(estado.get("violaciones", []), ensure_ascii=False))
    r = _llamar(cliente, modelo, _metodo=cliente.beta.messages.parse, max_tokens=8000, system=SISTEMA_AUDITOR,
                messages=[{"role": "user", "content": contenido}], output_format=Auditoria,
                output_config={"effort": "high"})
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return {"aprobado": False, "hallazgos": ["El auditor no pudo completar la revisión."]}
    return r.parsed_output.model_dump()


# ---------- grafo ----------

class Estado(TypedDict, total=False):
    solicitud: pd.DataFrame
    especialistas: Annotated[dict, operator.or_]
    decision: dict
    oferta: dict | None
    violaciones: list[str]
    auditoria: dict
    intentos: int
    final: dict
    historial: Annotated[list, operator.add]  # cada respuesta de cada nodo, para el registro


def cliente_anthropic():
    """Cliente real. Si hay LANGSMITH_API_KEY, cada llamada se traza en LangSmith (decisión 018)."""
    import anthropic

    cargar_env()
    cliente = anthropic.Anthropic()
    if os.environ.get("LANGSMITH_API_KEY"):
        from langsmith.wrappers import wrap_anthropic

        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "credit-risk-agent-lab")
        cliente = wrap_anthropic(cliente)
    return cliente


def construir(cliente=None, herr: Herramientas | None = None, modelo: str = MODELO):
    from langgraph.graph import END, START, StateGraph

    cliente = cliente or cliente_anthropic()
    herr = herr or Herramientas()

    def especialista(nombre):
        def nodo(estado):
            r = correr_especialista(cliente, herr, nombre, estado["solicitud"], modelo)
            return {"especialistas": {nombre: r}, "historial": [{"nodo": nombre, "intento": 1, "contenido": r}]}
        return nodo

    def comite(estado):
        d, n = decidir(cliente, estado, modelo), estado.get("intentos", 0) + 1
        return {"decision": d, "intentos": n, "historial": [{"nodo": "comite", "intento": n, "contenido": d}]}

    def reglas(estado):
        d = estado["decision"]
        oferta = None if d["decision"] == "rechazar" else herr.simular_oferta(estado["solicitud"], d["monto"], d["tasa"])
        return {"oferta": oferta, "violaciones": reglas_duras(d, oferta)}

    def auditor(estado):
        a = auditar(cliente, estado, modelo)
        if estado["violaciones"]:  # el código manda: con violaciones no hay aprobación
            a = {"aprobado": False, "hallazgos": a["hallazgos"] + estado["violaciones"]}
        contenido = a | {"oferta": estado["oferta"], "violaciones": estado["violaciones"]}
        return {"auditoria": a, "historial": [{"nodo": "auditor", "intento": estado["intentos"], "contenido": contenido}]}

    def siguiente(estado):
        if estado["auditoria"]["aprobado"] or estado["intentos"] >= INTENTOS:
            return "finalizar"
        return "comite"

    def finalizar(estado):
        d = dict(estado["decision"])
        if not estado["auditoria"]["aprobado"]:
            d = {"decision": "rechazar", "monto": 0.0, "tasa": 0.0,
                 "razones": ["La decisión no pasó la auditoría tras {} intentos.".format(estado["intentos"])]
                            + estado["auditoria"]["hallazgos"],
                 "memo": d.get("memo", "")}
        return {"final": d | {"auditoria": estado["auditoria"], "intentos": estado["intentos"]}}

    g = StateGraph(Estado)
    for nombre in ESPECIALISTAS:
        g.add_node(nombre, especialista(nombre))
        g.add_edge(START, nombre)
    g.add_node("comite", comite)
    g.add_node("reglas", reglas)
    g.add_node("auditor", auditor)
    g.add_node("finalizar", finalizar)
    g.add_edge(list(ESPECIALISTAS), "comite")
    g.add_edge("comite", "reglas")
    g.add_edge("reglas", "auditor")
    g.add_conditional_edges("auditor", siguiente, {"comite": "comite", "finalizar": "finalizar"})
    g.add_edge("finalizar", END)
    return g.compile()


def evaluar_solicitud(grafo, solicitud: pd.DataFrame, metadatos: dict | None = None) -> dict:
    config = {"run_name": "evaluar_solicitud", "metadata": metadatos or {}, "tags": [MODELO]}
    return grafo.invoke({"solicitud": solicitud, "especialistas": {}, "historial": []}, config=config)


def main() -> None:
    from credit_lab import registro
    from credit_lab.poblacion import cargar_poblacion

    p = argparse.ArgumentParser(description="Evalúa una solicitud de jul–dic 2015 con el agente 100% IA.")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--no-guardar", action="store_true", help="no registrar la ejecución en SQLite")
    p.add_argument("--modelo", default=MODELO, choices=sorted(PRECIOS))
    args = p.parse_args()
    df = cargar_poblacion()
    s2 = df[df["issue_d"] >= "2015-07-01"]
    sol = s2.sample(1, random_state=args.semilla)
    medidor = Medidor(cliente_anthropic(), args.modelo)
    inicio = time.perf_counter()
    estado = evaluar_solicitud(construir(medidor, modelo=args.modelo), sol, {"semilla": args.semilla})
    segundos = time.perf_counter() - inicio
    if not args.no_guardar:
        with registro.conectar() as con:
            n = registro.guardar(con, estado, medidor, segundos, semilla=args.semilla)
        print(f"Ejecución {n} registrada en {registro.RUTA}")
    final = estado["final"]
    print(final["memo"])
    print(f"\nDecisión: {final['decision']} · monto {final['monto']:,.0f} · tasa {final['tasa']:.2f}% · "
          f"auditoría {'aprobada' if final['auditoria']['aprobado'] else 'rechazada'} en {final['intentos']} intento(s)")
    print(f"Resultado real (no visto por el agente): {'malo' if sol['malo'].iloc[0] else 'bueno'}")


if __name__ == "__main__":
    main()

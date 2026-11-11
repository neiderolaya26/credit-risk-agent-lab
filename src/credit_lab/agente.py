"""Agente de crédito en LangGraph: el código decide con la política; la IA analiza, opina y redacta;
un analista firma todo lo que no sea una aprobación clara (decisiones 016, 021, 022 y 023).

    solicitud ─┬─ especialista de riesgo      (Claude + PD y razones TreeSHAP)
               ├─ especialista de economía    (Claude + prepago y pérdida)
               ├─ especialista de cupo/pricing (Claude + motor, estrés y simulación)
               └─ política (código: decisión y oferta, sin LLM)
                      ↓ (en paralelo)
               comité (Claude: opinión independiente, memo y qué verificar)
                      ↓
               reglas duras (código: cuota ≤ 20% del ingreso y ROI esperado ≥ 0% en la oferta)
                      ↓
               auditor (Claude: revisa el memo antes de que llegue al analista) ── observaciones → comité
                      ↓
               firma (automática solo si la política aprueba, el comité coincide y la auditoría pasa;
                      lo demás lo confirma o devuelve un analista, con interrupt de LangGraph)

Decisión 023 (issue #29): la política es determinista, así que la decisión y la oferta las calcula el
código (motor.decidir_politica) y coinciden con la política por construcción. El comité da su opinión;
si no coincide con la política, el caso va al analista marcado como desacuerdo. Las razones para el
cliente salen de un catálogo determinista (credit_lab.razones), no del memo.

Los números salen siempre de las herramientas (código sobre los modelos del repo). Si el auditor no
aprueba el memo tras INTENTOS rondas, el caso se escala a un analista (decisión 022).

Firma persistente (issue #26): el estado lleva la solicitud como dict (no un DataFrame) y el grafo usa un
checkpointer SQLite sin pickle (data/agente/firmas.db), así que un analista puede firmar después, desde otro
proceso, con el hilo de la solicitud.

Registro y observabilidad (decisión 018): cada ejecución se guarda en SQLite (credit_lab.registro) con el
resultado real del préstamo. Las trazas en LangSmith solo se activan con --trazar o CREDIT_LAB_TRAZAR=1.

Uso:
    python -m credit_lab.agente                         # un préstamo de jul–dic 2015; pide la firma en consola
    python -m credit_lab.agente --semilla 7 --pendiente # deja la propuesta pendiente e imprime su hilo
    python -m credit_lab.agente --firmar HILO --confirma --motivo confirma_politica --analista ana
Requiere ANTHROPIC_API_KEY (variable de entorno o archivo .env, que no se sube a git).
"""

import argparse
import copy
import json
import math
import operator
import os
import sqlite3
import threading
import time
import uuid
from typing import Annotated, Literal, TypedDict

import numpy as np
import pandas as pd
from pydantic import BaseModel

from credit_lab.causal import RANGO_OBSERVADO
from credit_lab.motor import CUOTA_MAX, ROI_EQUILIBRIO, ROI_OBJETIVO, TOPE_SUBIDA, Motor, decidir_politica
from credit_lab.poblacion import RAIZ

MODELO = "claude-sonnet-5-5"
VERSION = "v4"  # v0: piloto; v1: política compartida, auditor por materialidad; v2: memo sin invenciones; v3: regla 2b con ROI ≥ 4% (decisión 020); v4: el código decide, el comité opina (decisión 023)
# USD por millón de tokens (entrada, salida)
PRECIOS = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}
BETAS = ["server-side-fallback-2026-07-01"]
CON_FALLBACK = {"claude-sonnet-5-5", "claude-opus-5-5"}  # Haiku 5.5 no tiene fallback del servidor
INTENTOS = 2
MAX_TURNOS = 6
MONTO_MIN, MONTO_MAX = 1_000, 35_000
TASA_MIN, TASA_MAX = 5.0, 30.0
NO_OTORGA = {"rechazar", "escalar"}  # decisiones finales que no otorgan crédito (decisión 022)
RUTA_FIRMAS = RAIZ / "data" / "agente" / "firmas.db"
# motivo estructurado del analista (issue #38): lista cerrada, además del comentario libre
MOTIVOS_ANALISTA = {
    "confirma_politica": "confirma la propuesta",
    "datos_inconsistentes": "los datos de la solicitud no cuadran o falta verificarlos",
    "capacidad_pago": "duda sobre la capacidad de pago",
    "riesgo_no_capturado": "riesgo que el modelo no captura",
    "politica_comercial": "excepción comercial",
    "otro": "otro (ver comentario)",
}

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


def a_dict(sol: pd.DataFrame) -> dict:
    """La solicitud (una fila) como dict serializable sin pickle: las fechas van como texto ISO."""
    out = {}
    for k, v in sol.iloc[0].to_dict().items():
        if isinstance(v, pd.Timestamp):
            out[k] = {"__fecha__": v.isoformat()}
        elif isinstance(v, (np.floating, np.integer, np.bool_)):
            out[k] = v.item()
        else:
            out[k] = v
    # los tipos viajan con la fila: una columna de texto vacía volvería como número y rompería el codificador
    out["__tipos__"] = {k: str(t) for k, t in sol.dtypes.items() if not str(t).startswith("datetime")}
    return out


def a_df(sol) -> pd.DataFrame:
    """Reconstruye la fila que esperan los modelos a partir del dict del estado (o la deja si ya es DataFrame)."""
    if isinstance(sol, pd.DataFrame):
        return sol
    fila = {k: pd.Timestamp(v["__fecha__"]) if isinstance(v, dict) and "__fecha__" in v else v
            for k, v in sol.items() if k != "__tipos__"}
    df = pd.DataFrame([fila])
    for c, tipo in sol.get("__tipos__", {}).items():
        if c in df:
            df[c] = df[c].astype(tipo)
    return df


def resumen_solicitud(fila) -> dict:
    f = a_df(fila).iloc[0]
    r = {nombre: limpio(f.get(campo)) for campo, nombre in CAMPOS_SOLICITUD.items()}
    bajo, alto = f.get("fico_range_low"), f.get("fico_range_high")
    r[CAMPO_FICO] = limpio((bajo + alto) / 2 if alto is not None and pd.notna(alto) else bajo)
    return r


# ---------- herramientas: código puro sobre los modelos ----------

def validar_oferta(monto, tasa) -> tuple[float, float]:
    """Issue #27: el agente y el servidor MCP solo simulan dentro de la grilla del motor."""
    try:
        monto, tasa = float(monto), float(tasa)
    except (TypeError, ValueError):
        raise ValueError("monto y tasa deben ser números") from None
    if not (math.isfinite(monto) and MONTO_MIN <= monto <= MONTO_MAX):
        raise ValueError(f"monto fuera de rango: {monto:g}; use entre {MONTO_MIN:,} y {MONTO_MAX:,} USD")
    if not (math.isfinite(tasa) and TASA_MIN <= tasa <= TASA_MAX):
        raise ValueError(f"tasa fuera de rango: {tasa:g}; use entre {TASA_MIN:g}% y {TASA_MAX:g}% anual")
    return monto, tasa


class Herramientas:
    def __init__(self, motor: Motor | None = None):
        from credit_lab.explicar import contribuciones, explicar_prestamo

        self.motor = motor or Motor()
        self.motor_estres = copy.copy(self.motor)
        self.motor_estres.estres = True
        self._explicar, self._contribuciones = explicar_prestamo, contribuciones

    def riesgo(self, sol: pd.DataFrame) -> dict:
        from credit_lab.razones import PROXIES_EDAD

        r = self._explicar(self.motor.campeon, sol, top=8)
        # issue #39: los proxies de edad no se usan como razón, ni en el memo ni ante el cliente
        razones = [x for x in r["razones"] if x["variable"] not in PROXIES_EDAD][:5]
        return limpio({"pd": r["pd"], "pd_promedio_cartera": r["pd_base"],
                       "razones": [{k: x[k] for k in ("descripcion", "valor", "efecto", "contribucion_logit")}
                                   for x in razones]})

    def razones_cliente(self, sol: pd.DataFrame) -> list[dict]:
        from credit_lab.razones import razones_cliente

        c = self._contribuciones(self.motor.campeon, sol).iloc[0].drop("valor_base")
        return razones_cliente(c)

    def economia(self, sol: pd.DataFrame) -> dict:
        ev = self.motor.evaluar(sol).iloc[0]
        p = self.motor.supervivencia.distribucion(sol)[0]
        return limpio({"r_bueno": ev["r_bueno"], "r_malo": ev["r_malo"], "pd": ev["pd"],
                       "roi_esperado": ev["roi_esperado"], "utilidad_esperada": ev["utilidad_esperada"],
                       "prob_prepago_antes_mes_30": p[:29].sum(),
                       "mes_esperado_de_pago": float((np.arange(1, 37) * p).sum())})

    def politica(self, sol: pd.DataFrame) -> dict:
        """La decisión de la política, sin LLM (decisión 023)."""
        return limpio(decidir_politica(self.motor, sol))

    def cupo_pricing(self, sol: pd.DataFrame) -> dict:
        base, estres = self.motor.recomendar(sol), self.motor_estres.recomendar(sol)
        # las dos contraofertas posibles, ya simuladas, para que se puedan verificar
        contraofertas = {}
        if 0 < base["monto_maximo_objetivo"] < base["monto_pedido"]:
            contraofertas["contraoferta_b_monto_objetivo"] = self.simular_oferta(sol, base["monto_maximo_objetivo"], base["tasa_ofrecida"])
        if pd.notna(base["tasa_sugerida"]):
            contraofertas["contraoferta_a_tasa_sugerida"] = self.simular_oferta(sol, base["monto_pedido"], base["tasa_sugerida"])
        return limpio(base | contraofertas | {"tasa_equilibrio_estres": estres["tasa_equilibrio"],
                              "tasa_sugerida_estres": estres["tasa_sugerida"],
                              "regla_cuota_max": CUOTA_MAX, "tope_subida_tasa_pts": TOPE_SUBIDA} | CRITERIOS)

    def simular_oferta(self, sol: pd.DataFrame, monto: float, tasa: float) -> dict:
        monto, tasa = validar_oferta(monto, tasa)
        s = sol.copy()
        s["loan_amnt"], s["int_rate"] = monto, tasa
        ev = self.motor.evaluar(s).iloc[0]
        return limpio({"monto": monto, "tasa": tasa, "pd": ev["pd"], "roi_esperado": ev["roi_esperado"],
                       "utilidad_esperada": ev["utilidad_esperada"], "cuota_ingreso": ev["cuota_ingreso"],
                       "cumple_capacidad": ev["cumple_capacidad"], "tasa_mercado": ev["tasa_mercado"],
                       "fuera_de_rango": ev["fuera_de_rango"]} | CRITERIOS)


SIN_ARGUMENTOS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
ESQUEMA_SIMULAR = {
    "type": "object",
    "properties": {"monto": {"type": "number", "description": f"Monto en USD ({MONTO_MIN:,} a {MONTO_MAX:,})"},
                   "tasa": {"type": "number", "description": f"Tasa anual en % ({TASA_MIN:g} a {TASA_MAX:g})"}},
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
                   "(supuesto: la PD sube 1 pt por punto de sobreprecio) y la bandera fuera de rango (> 1 pt de la "
                   "tasa de mercado: sin evidencia). Puedes simular ofertas alternativas. Propón la mejor oferta.",
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

class Opinion(BaseModel):
    """Lo que devuelve el comité: su opinión independiente y el memo; la decisión la toma la política."""
    opinion: Literal["aprobar", "rechazar", "contraoferta"]
    razones: list[str]
    que_verificar: list[str]
    memo: str


class Auditoria(BaseModel):
    aprobado: bool
    hallazgos: list[str]


POLITICA = f"""Política del banco (en este orden):
1. Aprobar si, al monto y la tasa pedidos, el ROI esperado es ≥ {ROI_OBJETIVO:.0%} y la cuota es ≤ {CUOTA_MAX:.0%} del ingreso.
2. Si no, contraoferta con la primera opción que cumpla (primero se baja el monto, después se sube la tasa):
   b. el monto máximo objetivo (el mayor monto con ROI ≥ {ROI_OBJETIVO:.0%} y cuota ≤ {CUOTA_MAX:.0%} a la tasa ofrecida), si es menor que el pedido;
   a. el monto pedido a la tasa sugerida (ROI ≥ {ROI_OBJETIVO:.0%}), si sube como máximo {TOPE_SUBIDA:g} puntos sobre la tasa
      ofrecida, no queda fuera de rango frente a la tasa de mercado y la cuota sigue ≤ {CUOTA_MAX:.0%}.
3. Si no, rechazar (monto = 0, tasa = 0).
Reglas duras que verifica el código: cuota ≤ {CUOTA_MAX:.0%} del ingreso y ROI esperado ≥ {ROI_EQUILIBRIO:.0%} en lo que se otorga.
El escenario de estrés es un supuesto, no una estimación: se menciona como alerta.
El FICO de la solicitud es el punto medio del rango, el mismo que usa el modelo."""

SISTEMA_COMITE = f"""Eres el comité de crédito. La decisión y la oferta las calcula el código con esta política:

{POLITICA}

Tú recibes esa decisión, el análisis de tres especialistas y los resultados de sus herramientas. Tu trabajo:
1. Dar tu opinión independiente (aprobar, contraoferta o rechazar) según la política y los datos. Si no coincide
   con la decisión de la política, explica en razones por qué; el caso irá a un analista marcado como desacuerdo.
2. Listar en que_verificar lo que el analista debería revisar antes de firmar (datos que no cuadran entre la
   solicitud y las herramientas, alertas de estrés o fuera de rango, capacidad de pago ajustada). Lista vacía si
   no hay nada.
3. Redactar el memo para el analista, en español y en markdown, con secciones: Resumen, Decisión de la política,
   Opinión del comité, Riesgo, Economía, Cupo y precio, Alertas, Qué verificar.
Cita solo números que aparecen en los resultados de las herramientas, en la decisión de la política o en la
política. No agregues explicaciones causales que no salgan de las herramientas. Atribuye a cada especialista
solo lo que dijo. Antes de afirmar que un dato no existe, revisa los resultados de las herramientas."""

SISTEMA_AUDITOR = f"""Eres el auditor de crédito: revisas la propuesta y el memo antes de que lleguen al analista,
que es quien firma. La política vigente es:

{POLITICA}

Verifica que:
1. Cada número del memo coincide (con redondeo) con los resultados de las herramientas, con la decisión de la
   política o con la política.
2. El memo describe correctamente la decisión de la política y la opinión del comité.
3. No hay datos inventados ni contradicciones entre memo, razones y opinión.
Rechaza solo por problemas materiales: un número incorrecto, una decisión mal descrita o un dato inventado que
sostenga la opinión. Las observaciones menores (redacción, énfasis) van en hallazgos con el prefijo "Menor:" y no
impiden aprobar."""


# ---------- reglas duras ----------

def reglas_duras(decision: dict, oferta: dict | None) -> list[str]:
    """Reglas que nada puede saltarse. Devuelve la lista de violaciones de lo que se otorga."""
    if decision["decision"] in NO_OTORGA:
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
    llamadas, agotado = [], True
    for _ in range(MAX_TURNOS):
        r = _llamar(cliente, modelo, _metodo=cliente.beta.messages.create, max_tokens=8000,
                    system=cfg["sistema"], tools=tools, messages=mensajes,
                    output_config={"effort": "medium"})
        if r.stop_reason == "refusal":
            return {"analisis": "El modelo declinó analizar esta solicitud.", "herramientas": llamadas, "agotado": False}
        mensajes.append({"role": "assistant", "content": r.content})
        usos = [b for b in r.content if b.type == "tool_use"]
        if not usos:
            agotado = False
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
    analisis = _texto(r)
    if agotado:  # issue #31: el comité tiene que saber que este análisis quedó incompleto
        analisis = (analisis + "\n\n" if analisis else "") + f"[Análisis incompleto: se agotaron los {MAX_TURNOS} turnos.]"
    return {"analisis": analisis, "herramientas": llamadas, "agotado": agotado}


def _contexto(estado) -> str:
    return json.dumps({"solicitud": resumen_solicitud(estado["solicitud"]),
                       "decision_de_la_politica": estado["politica"],
                       "especialistas": estado["especialistas"]}, ensure_ascii=False, indent=1)


def opinar(cliente, estado, modelo: str = MODELO) -> dict:
    partes = [_contexto(estado)]
    if estado.get("auditoria") and not estado["auditoria"]["aprobado"]:
        partes.append("El auditor rechazó tu memo anterior. Corrige estos hallazgos:\n- "
                      + "\n- ".join(estado["auditoria"]["hallazgos"]))
    # messages.parse con un modelo Pydantic es el uso documentado del SDK; lo obsoleto es output_format en
    # messages.create (issue #32)
    r = _llamar(cliente, modelo, _metodo=cliente.beta.messages.parse, max_tokens=16000, system=SISTEMA_COMITE,
                messages=[{"role": "user", "content": "\n\n".join(partes)}], output_format=Opinion,
                output_config={"effort": "high"})
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return {"opinion": None, "razones": ["El comité no pudo emitir una opinión."], "que_verificar": [], "memo": ""}
    return r.parsed_output.model_dump()


def auditar(cliente, estado, modelo: str = MODELO) -> dict:
    contenido = (_contexto(estado) + "\n\nOpinión y memo del comité:\n"
                 + json.dumps(estado["comite"], ensure_ascii=False, indent=1)
                 + "\n\nOferta de la política recalculada con el motor:\n" + json.dumps(estado.get("oferta"), ensure_ascii=False))
    r = _llamar(cliente, modelo, _metodo=cliente.beta.messages.parse, max_tokens=8000, system=SISTEMA_AUDITOR,
                messages=[{"role": "user", "content": contenido}], output_format=Auditoria,
                output_config={"effort": "high"})
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return {"aprobado": False, "hallazgos": ["El auditor no pudo completar la revisión."]}
    return r.parsed_output.model_dump()


# ---------- grafo ----------

class Estado(TypedDict, total=False):
    solicitud: dict  # dict y no DataFrame: el estado se guarda sin pickle (issue #26)
    hilo: str
    especialistas: Annotated[dict, operator.or_]
    politica: dict
    comite: dict
    oferta: dict | None
    violaciones: list[str]
    auditoria: dict
    intentos: int
    final: dict
    historial: Annotated[list, operator.add]  # cada respuesta de cada nodo, para el registro


def cliente_anthropic(trazar: bool | None = None):
    """Cliente real. Las trazas en LangSmith solo se activan si se piden (issue #28): trazar=True,
    o CREDIT_LAB_TRAZAR=1, y además debe existir LANGSMITH_API_KEY."""
    import anthropic

    cargar_env()
    cliente = anthropic.Anthropic()
    if trazar is None:
        trazar = os.environ.get("CREDIT_LAB_TRAZAR", "").lower() in {"1", "true", "si", "sí"}
    if trazar and os.environ.get("LANGSMITH_API_KEY"):
        from langsmith.wrappers import wrap_anthropic

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", "credit-risk-agent-lab")
        cliente = wrap_anthropic(cliente)
    return cliente


def checkpointer_sqlite(ruta=RUTA_FIRMAS):
    """Checkpointer persistente y sin pickle: la firma sobrevive al proceso (issue #26)."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    ruta.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(ruta, check_same_thread=False), serde=JsonPlusSerializer())


def construir(cliente=None, herr: Herramientas | None = None, modelo: str = MODELO, firma_humana: bool = True,
              checkpointer=None):
    """firma_humana=True agrega el nodo de firma y un checkpointer (por defecto SQLite en data/agente/firmas.db)."""
    from langgraph.graph import END, START, StateGraph

    cliente = cliente or cliente_anthropic()
    herr = herr or Herramientas()

    def especialista(nombre):
        def nodo(estado):
            r = correr_especialista(cliente, herr, nombre, a_df(estado["solicitud"]), modelo)
            return {"especialistas": {nombre: r}, "historial": [{"nodo": nombre, "intento": 1, "contenido": r}]}
        return nodo

    def politica(estado):
        p = herr.politica(a_df(estado["solicitud"]))
        return {"politica": p, "historial": [{"nodo": "politica", "intento": 1, "contenido": p}]}

    def comite(estado):
        c, n = opinar(cliente, estado, modelo), estado.get("intentos", 0) + 1
        return {"comite": c, "intentos": n, "historial": [{"nodo": "comite", "intento": n, "contenido": c}]}

    def reglas(estado):
        p = estado["politica"]
        oferta = None if p["decision"] in NO_OTORGA else herr.simular_oferta(a_df(estado["solicitud"]), p["monto"], p["tasa"])
        return {"oferta": oferta, "violaciones": reglas_duras(p, oferta)}

    def auditor(estado):
        a = auditar(cliente, estado, modelo)
        contenido = a | {"oferta": estado["oferta"], "violaciones": estado["violaciones"]}
        return {"auditoria": a, "historial": [{"nodo": "auditor", "intento": estado["intentos"], "contenido": contenido}]}

    def siguiente(estado):
        if estado["auditoria"]["aprobado"] or estado["intentos"] >= INTENTOS:
            return "finalizar"
        return "comite"

    def finalizar(estado):
        p, c = estado["politica"], estado["comite"]
        d = {"decision": p["decision"], "monto": p["monto"], "tasa": p["tasa"], "regla": p["regla"]}
        desacuerdo = c.get("opinion") != p["decision"]
        escalar = not estado["auditoria"]["aprobado"] or bool(estado["violaciones"])
        if escalar:  # el memo no se pudo validar: el caso se escala con la propuesta de la política (decisión 022)
            d = {"decision": "escalar", "monto": p["monto"], "tasa": p["tasa"], "regla": p["regla"],
                 "propuesta_original": {k: p[k] for k in ("decision", "monto", "tasa")}}
        automatica = d["decision"] == "aprobar" and not desacuerdo and not escalar
        razones_cliente, carta = [], ""
        if p["decision"] in {"rechazar", "contraoferta"}:  # issue #37: catálogo determinista, separado del memo
            from credit_lab.razones import carta as redactar_carta

            razones_cliente = herr.razones_cliente(a_df(estado["solicitud"]))
            carta = redactar_carta(p["decision"], razones_cliente, p)
        estado_final = "aprobada automáticamente" if automatica else "pendiente de firma"
        if not firma_humana:
            estado_final = "propuesta sin firma (eval)"
        final = d | {"decision_politica": p["decision"], "opinion_comite": c.get("opinion"), "desacuerdo": desacuerdo,
                     "razones": c.get("razones", []), "que_verificar": c.get("que_verificar", []),
                     "memo": c.get("memo", ""), "razones_cliente": razones_cliente, "carta_cliente": carta,
                     "auditoria": estado["auditoria"], "intentos": estado["intentos"],
                     "violaciones": estado["violaciones"], "estado": estado_final,
                     "decision_final": d["decision"] if automatica else None}
        return {"final": final}

    def firma(estado):
        from langgraph.types import interrupt

        final = estado["final"]
        if final["estado"] == "aprobada automáticamente":
            return {}
        respuesta = interrupt({"propuesta": {k: final[k] for k in ("decision", "monto", "tasa", "regla")},
                               "decision_politica": final["decision_politica"],
                               "opinion_comite": final["opinion_comite"], "desacuerdo": final["desacuerdo"],
                               "que_verificar": final["que_verificar"], "memo": final["memo"],
                               "motivos_validos": sorted(MOTIVOS_ANALISTA)})
        respuesta = validar_firma(respuesta)
        efectiva = final["propuesta_original"]["decision"] if final["decision"] == "escalar" else final["decision"]
        return {"final": final | {"estado": "firmada por analista" if respuesta["confirma"] else "devuelta por analista",
                                  "firma": respuesta,
                                  "decision_final": efectiva if respuesta["confirma"] else None},
                "historial": [{"nodo": "firma", "intento": 1, "contenido": respuesta}]}

    g = StateGraph(Estado)
    for nombre in ESPECIALISTAS:
        g.add_node(nombre, especialista(nombre))
        g.add_edge(START, nombre)
    g.add_node("politica", politica)
    g.add_edge(START, "politica")
    g.add_node("comite", comite)
    g.add_node("reglas", reglas)
    g.add_node("auditor", auditor)
    g.add_node("finalizar", finalizar)
    g.add_edge([*ESPECIALISTAS, "politica"], "comite")
    g.add_edge("comite", "reglas")
    g.add_edge("reglas", "auditor")
    g.add_conditional_edges("auditor", siguiente, {"comite": "comite", "finalizar": "finalizar"})
    if not firma_humana:
        g.add_edge("finalizar", END)
        return g.compile()
    g.add_node("firma", firma)
    g.add_edge("finalizar", "firma")
    g.add_edge("firma", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else checkpointer_sqlite())


def validar_firma(respuesta) -> dict:
    """Issue #26: el valor con que se retoma el grafo debe ser una firma bien formada."""
    if not isinstance(respuesta, dict) or not isinstance(respuesta.get("confirma"), bool):
        raise ValueError("la firma debe ser un dict con 'confirma' (bool)")
    analista = str(respuesta.get("analista") or "").strip()
    if not analista:
        raise ValueError("la firma debe indicar el analista")
    motivo = respuesta.get("motivo") or ("confirma_politica" if respuesta["confirma"] else None)
    if motivo not in MOTIVOS_ANALISTA:
        raise ValueError(f"motivo inválido: {motivo!r}; use uno de {sorted(MOTIVOS_ANALISTA)}")
    return {"confirma": respuesta["confirma"], "analista": analista, "motivo": motivo,
            "comentario": str(respuesta.get("comentario") or ""), "fecha": pd.Timestamp.now(tz="UTC").isoformat()}


def _config(metadatos: dict | None, hilo: str) -> dict:
    return {"run_name": "evaluar_solicitud", "metadata": metadatos or {}, "tags": [MODELO],
            "configurable": {"thread_id": hilo}}


def evaluar_solicitud(grafo, solicitud: pd.DataFrame, metadatos: dict | None = None, hilo: str | None = None) -> dict:
    """Corre el grafo con un hilo propio por solicitud (issue #25). Si el caso espera firma, devuelve el estado
    con final['estado'] = 'pendiente de firma', la propuesta en estado['__interrupt__'] y el hilo en
    estado['hilo'] para completarlo con firmar()."""
    hilo = hilo or uuid.uuid4().hex
    return grafo.invoke({"solicitud": a_dict(solicitud), "hilo": hilo, "especialistas": {}, "historial": []},
                        config=_config(metadatos, hilo))


def firmar(grafo, hilo: str, confirma: bool, analista: str, motivo: str | None = None, comentario: str = "") -> dict:
    """Retoma el grafo pausado con la firma del analista. Funciona desde otro proceso con el checkpointer SQLite."""
    from langgraph.types import Command

    respuesta = {"confirma": confirma, "analista": analista, "motivo": motivo, "comentario": comentario}
    validar_firma(respuesta)
    return grafo.invoke(Command(resume=respuesta), config={"configurable": {"thread_id": hilo}})


def main() -> None:
    from credit_lab import registro
    from credit_lab.poblacion import cargar_poblacion

    p = argparse.ArgumentParser(description="Evalúa una solicitud de jul–dic 2015 con el agente de crédito.")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--no-guardar", action="store_true", help="no registrar la ejecución en SQLite")
    p.add_argument("--modelo", default=MODELO, choices=sorted(PRECIOS))
    p.add_argument("--trazar", action="store_true", help="enviar trazas a LangSmith (requiere LANGSMITH_API_KEY)")
    p.add_argument("--pendiente", action="store_true", help="no pedir la firma ahora: imprime el hilo para firmar después")
    p.add_argument("--firmar", metavar="HILO", help="firma una propuesta pendiente")
    p.add_argument("--confirma", action="store_true", help="con --firmar: confirma la propuesta (si no, la devuelve)")
    p.add_argument("--motivo", choices=sorted(MOTIVOS_ANALISTA))
    p.add_argument("--analista", default="analista")
    p.add_argument("--comentario", default="")
    args = p.parse_args()

    medidor = Medidor(cliente_anthropic(args.trazar), args.modelo)
    if args.firmar:  # firma diferida: retoma el grafo guardado en data/agente/firmas.db
        grafo = construir(medidor, Herramientas(), modelo=args.modelo)
        estado = firmar(grafo, args.firmar, args.confirma, args.analista, args.motivo, args.comentario)
        if not args.no_guardar:
            with registro.conectar() as con:
                registro.actualizar_firma(con, args.firmar, estado["final"])
        print(f"{estado['final']['estado']} · decisión final: {estado['final']['decision_final']}")
        return

    df = cargar_poblacion()
    s2 = df[df["issue_d"] >= "2015-07-01"]
    sol = s2.sample(1, random_state=args.semilla)
    inicio = time.perf_counter()
    grafo = construir(medidor, modelo=args.modelo)
    estado = evaluar_solicitud(grafo, sol, {"semilla": args.semilla})
    segundos = time.perf_counter() - inicio
    hilo = estado["hilo"]
    if not args.no_guardar:
        with registro.conectar() as con:
            n = registro.guardar(con, estado, medidor, segundos, semilla=args.semilla, solicitud=sol)
        print(f"Ejecución {n} registrada en {registro.RUTA}")
    final = estado["final"]
    print(final["memo"])
    print(f"\nPolítica: {final['decision_politica']} (regla {final['regla']}) · opinión del comité: "
          f"{final['opinion_comite']}{' · DESACUERDO' if final['desacuerdo'] else ''}")
    print(f"Propuesta: {final['decision']} · monto {final['monto']:,.0f} · tasa {final['tasa']:.2f}% · {final['estado']}")
    if final["estado"] == "pendiente de firma":
        if args.pendiente:
            print(f"Pendiente de firma. Para firmar: python -m credit_lab.agente --firmar {hilo} --confirma --analista <nombre>")
            return
        confirma = input("¿Firma la propuesta? [s/n]: ").strip().lower().startswith("s")
        motivo = input(f"Motivo ({', '.join(sorted(MOTIVOS_ANALISTA))}): ").strip() or None
        comentario = input("Comentario del analista (opcional): ").strip()
        estado = firmar(grafo, hilo, confirma, analista=args.analista, motivo=motivo, comentario=comentario)
        if not args.no_guardar:
            with registro.conectar() as con:
                registro.actualizar_firma(con, hilo, estado["final"])
        print(f"{estado['final']['estado']} · decisión final: {estado['final']['decision_final']}")
    print(f"Resultado real (no visto por el agente): {'malo' if sol['malo'].iloc[0] else 'bueno'}")


if __name__ == "__main__":
    main()

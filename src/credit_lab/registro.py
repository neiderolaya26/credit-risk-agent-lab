"""Registro de ejecuciones del agente en SQLite (decisión 018).

Cada solicitud evaluada deja:
- una fila en `ejecuciones`: decisión, auditoría, tokens, costo, latencia y el
  **resultado real** del préstamo (malo y ROI), que el agente nunca vio;
- una fila por respuesta en `pasos`: cada especialista (con sus llamadas a
  herramientas), cada decisión del comité y cada auditoría.

Así se puede evaluar después no solo si el agente redacta bien, sino si sus
decisiones rinden: ROI real y mora de lo que aprueba frente a lo que rechaza.
La base vive en data/agente/ y no se sube a git.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from credit_lab.agente import MODELO, a_df, limpio, resumen_solicitud
from credit_lab.poblacion import RAIZ
from credit_lab.prepago import rendimiento

RUTA = RAIZ / "data" / "agente" / "ejecuciones.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS ejecuciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    modelo TEXT NOT NULL,
    semilla INTEGER,
    solicitud TEXT NOT NULL,
    decision TEXT NOT NULL,
    monto REAL,
    tasa REAL,
    auditoria_aprobada INTEGER NOT NULL,
    intentos INTEGER NOT NULL,
    hallazgos TEXT,
    memo TEXT,
    llamadas_llm INTEGER,
    tokens_entrada INTEGER,
    tokens_salida INTEGER,
    costo_usd REAL,
    segundos REAL,
    malo_real INTEGER,
    roi_real REAL,
    estado TEXT
);
CREATE TABLE IF NOT EXISTS pasos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ejecucion_id INTEGER NOT NULL REFERENCES ejecuciones(id),
    orden INTEGER NOT NULL,
    nodo TEXT NOT NULL,
    intento INTEGER NOT NULL,
    contenido TEXT NOT NULL
);
"""
# columnas agregadas después (decisiones 021 y 023): se crean con ALTER TABLE en bases viejas
COLUMNAS_NUEVAS = {
    "estado": "TEXT", "hilo": "TEXT", "decision_politica": "TEXT", "opinion_comite": "TEXT",
    "desacuerdo": "INTEGER", "decision_final": "TEXT", "analista": "TEXT", "motivo": "TEXT",
    "comentario": "TEXT", "fecha_firma": "TEXT",
}


@contextmanager
def conectar(ruta: Path = RUTA) -> Iterator[sqlite3.Connection]:
    """Abre la base (creando las tablas y columnas que falten) y la cierra al salir del with."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta)
    try:
        con.executescript(ESQUEMA)
        columnas = {c[1] for c in con.execute("PRAGMA table_info(ejecuciones)")}
        for nombre, tipo in COLUMNAS_NUEVAS.items():
            if nombre not in columnas:
                con.execute(f"ALTER TABLE ejecuciones ADD COLUMN {nombre} {tipo}")
        yield con
    finally:
        con.close()


def _json(x) -> str:
    return json.dumps(limpio(x), ensure_ascii=False)


def guardar(con: sqlite3.Connection, estado: dict, medidor=None, segundos: float | None = None,
            semilla: int | None = None, modelo: str | None = None, solicitud: pd.DataFrame | None = None) -> int:
    """Guarda una ejecución y sus pasos; devuelve el id. `decision` es la propuesta; `decision_final` es la
    que cuenta (issue #18): queda vacía hasta que la firma un analista, y vacía si la devuelve."""
    sol, final = a_df(solicitud if solicitud is not None else estado["solicitud"]), estado["final"]
    malo = int(sol["malo"].iloc[0]) if "malo" in sol else None
    roi = float(rendimiento(sol).iloc[0]) if {"total_pymnt", "funded_amnt", "collection_recovery_fee"} <= set(sol) else None
    firma = final.get("firma") or {}
    fila = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": modelo or getattr(medidor, "modelo", None) or MODELO, "semilla": semilla,
        "solicitud": _json(resumen_solicitud(sol)), "decision": final["decision"], "monto": final["monto"],
        "tasa": final["tasa"], "auditoria_aprobada": int(final["auditoria"]["aprobado"]), "intentos": final["intentos"],
        "hallazgos": _json(final["auditoria"]["hallazgos"]), "memo": final.get("memo", ""),
        "llamadas_llm": getattr(medidor, "llamadas", None), "tokens_entrada": getattr(medidor, "tokens_entrada", None),
        "tokens_salida": getattr(medidor, "tokens_salida", None), "costo_usd": medidor.costo_usd() if medidor else None,
        "segundos": segundos, "malo_real": malo, "roi_real": roi, "estado": final.get("estado"),
        "hilo": estado.get("hilo"), "decision_politica": final.get("decision_politica"),
        "opinion_comite": final.get("opinion_comite"),
        "desacuerdo": None if final.get("desacuerdo") is None else int(final["desacuerdo"]),
        "decision_final": final.get("decision_final"), "analista": firma.get("analista"),
        "motivo": firma.get("motivo"), "comentario": firma.get("comentario"), "fecha_firma": firma.get("fecha"),
    }
    with con:
        cur = con.execute(f"INSERT INTO ejecuciones ({', '.join(fila)}) VALUES ({', '.join('?' * len(fila))})",
                          tuple(fila.values()))
        ejecucion = cur.lastrowid
        con.executemany(
            "INSERT INTO pasos (ejecucion_id, orden, nodo, intento, contenido) VALUES (?, ?, ?, ?, ?)",
            [(ejecucion, i, p["nodo"], p["intento"], _json(p["contenido"]))
             for i, p in enumerate(estado.get("historial", []))])
    return ejecucion


def actualizar_firma(con: sqlite3.Connection, hilo: str, final: dict) -> None:
    """Registra la firma diferida de un analista en la ejecución de ese hilo."""
    firma = final.get("firma") or {}
    with con:
        con.execute("UPDATE ejecuciones SET estado = ?, decision_final = ?, analista = ?, motivo = ?, comentario = ?,"
                    " fecha_firma = ? WHERE hilo = ?",
                    (final.get("estado"), final.get("decision_final"), firma.get("analista"), firma.get("motivo"),
                     firma.get("comentario"), firma.get("fecha"), hilo))


def leer(con: sqlite3.Connection, tabla: str = "ejecuciones") -> pd.DataFrame:
    if tabla not in {"ejecuciones", "pasos"}:
        raise ValueError(f"tabla desconocida: {tabla}")
    return pd.read_sql_query(f"SELECT * FROM {tabla}", con)


def resumen(con: sqlite3.Connection) -> pd.DataFrame:
    """Desempeño por decisión efectiva (issue #18): solo cuentan las aprobadas automáticamente y las firmadas.
    Las pendientes y las devueltas por el analista no son decisiones y quedan fuera."""
    e = leer(con)
    if e.empty:
        return e
    e = e[e["decision_final"].notna()]
    return e.groupby(["decision_final", "estado"]).agg(
        ejecuciones=("id", "size"),
        mora_real=("malo_real", "mean"),
        roi_real=("roi_real", "mean"),
        auditoria_aprobada=("auditoria_aprobada", "mean"),
        intentos=("intentos", "mean"),
        costo_usd=("costo_usd", "mean"),
        segundos=("segundos", "mean"),
    )


def kpis(con: sqlite3.Connection) -> dict:
    """KPIs de la cola del analista (issue #38)."""
    e = leer(con)
    if e.empty:
        return {}
    firmadas = e["estado"].isin(["firmada por analista", "devuelta por analista"])
    t_firma = (pd.to_datetime(e["fecha_firma"], utc=True, format="ISO8601")
               - pd.to_datetime(e["fecha"], utc=True, format="ISO8601")).dt.total_seconds() / 60
    return {
        "ejecuciones": len(e),
        "aprobadas_automaticamente": e["estado"].eq("aprobada automáticamente").mean(),
        "a_firma": e["estado"].ne("aprobada automáticamente").mean(),
        "pendientes": e["estado"].eq("pendiente de firma").mean(),
        "devueltas_sobre_firmadas": e.loc[firmadas, "estado"].eq("devuelta por analista").mean() if firmadas.any() else None,
        "desacuerdos_comite_politica": e["desacuerdo"].mean(),
        "motivos_analista": e.loc[firmadas, "motivo"].value_counts().to_dict(),
        "minutos_hasta_firma": t_firma[firmadas].median() if firmadas.any() else None,
    }

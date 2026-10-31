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

from credit_lab.agente import MODELO, limpio, resumen_solicitud
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


@contextmanager
def conectar(ruta: Path = RUTA) -> Iterator[sqlite3.Connection]:
    """Abre la base (creando las tablas si faltan) y la cierra al salir del with."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta)
    try:
        con.executescript(ESQUEMA)
        columnas = {c[1] for c in con.execute("PRAGMA table_info(ejecuciones)")}
        if "estado" not in columnas:  # bases creadas antes de la decisión 021
            con.execute("ALTER TABLE ejecuciones ADD COLUMN estado TEXT")
        yield con
    finally:
        con.close()


def _json(x) -> str:
    return json.dumps(limpio(x), ensure_ascii=False)


def guardar(con: sqlite3.Connection, estado: dict, medidor=None, segundos: float | None = None,
            semilla: int | None = None, modelo: str | None = None) -> int:
    """Guarda una ejecución y sus pasos; devuelve el id. Sin la columna `malo`, el resultado real queda vacío."""
    sol, final = estado["solicitud"], estado["final"]
    malo = int(sol["malo"].iloc[0]) if "malo" in sol else None
    roi = float(rendimiento(sol).iloc[0]) if {"total_pymnt", "funded_amnt", "collection_recovery_fee"} <= set(sol) else None
    with con:
        cur = con.execute(
            "INSERT INTO ejecuciones (fecha, modelo, semilla, solicitud, decision, monto, tasa, auditoria_aprobada,"
            " intentos, hallazgos, memo, llamadas_llm, tokens_entrada, tokens_salida, costo_usd, segundos,"
            " malo_real, roi_real, estado) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             modelo or getattr(medidor, "modelo", None) or MODELO, semilla,
             _json(resumen_solicitud(sol)), final["decision"], final["monto"], final["tasa"],
             int(final["auditoria"]["aprobado"]), final["intentos"], _json(final["auditoria"]["hallazgos"]),
             final.get("memo", ""),
             getattr(medidor, "llamadas", None), getattr(medidor, "tokens_entrada", None),
             getattr(medidor, "tokens_salida", None), medidor.costo_usd() if medidor else None,
             segundos, malo, roi, final.get("estado")))
        ejecucion = cur.lastrowid
        con.executemany(
            "INSERT INTO pasos (ejecucion_id, orden, nodo, intento, contenido) VALUES (?, ?, ?, ?, ?)",
            [(ejecucion, i, p["nodo"], p["intento"], _json(p["contenido"]))
             for i, p in enumerate(estado.get("historial", []))])
    return ejecucion


def leer(con: sqlite3.Connection, tabla: str = "ejecuciones") -> pd.DataFrame:
    if tabla not in {"ejecuciones", "pasos"}:
        raise ValueError(f"tabla desconocida: {tabla}")
    return pd.read_sql_query(f"SELECT * FROM {tabla}", con)


def resumen(con: sqlite3.Connection) -> pd.DataFrame:
    """Desempeño por decisión: cuántas, mora y ROI real, auditoría al primer intento, costo y latencia."""
    e = leer(con)
    if e.empty:
        return e
    return e.groupby("decision").agg(
        ejecuciones=("id", "size"),
        mora_real=("malo_real", "mean"),
        roi_real=("roi_real", "mean"),
        auditoria_aprobada=("auditoria_aprobada", "mean"),
        intentos=("intentos", "mean"),
        costo_usd=("costo_usd", "mean"),
        segundos=("segundos", "mean"),
    )

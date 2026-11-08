"""Runner de las evals del agente (decisión 019).

Por variante (modelo del agente) escribe en evals/agente/<modelo>/:
- results.jsonl: una fila por caso, escrita apenas termina (se puede reanudar sin duplicar)
- traces/<id>.json: historial completo del grafo (especialistas, comité, auditor) y veredicto del juez
- errors.jsonl: intentos fallidos (no ocupan el lugar del caso, así que se reintentan al reanudar)
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from credit_lab import registro
import hashlib

from credit_lab.agente import (PRECIOS, VERSION, Herramientas, Medidor, a_df, cliente_anthropic, construir,
                               evaluar_solicitud, NO_OTORGA, limpio, reglas_duras, resumen_solicitud)
from credit_lab.evals import CARPETA, JUEZ, RUBRICA, cargar_muestra, fidelidad, juzgar, piso_fidelidad
from credit_lab.prepago import rendimiento

HILOS = 4
# qué versión de la política sirve de referencia para cada versión del agente (issue #19): las versiones se
# comparan con la política con la que se construyeron, y el reporte dice cuál es
POLITICA_DE_VERSION = {"v0": "020", "v1": "020", "v2": "020", "v3": "020", "v4": "023"}
CAMPO_REFERENCIA = {"020": "decision_motor_020", "023": "decision_motor"}


def referencia(caso: dict, version: str) -> str:
    return caso.get(CAMPO_REFERENCIA[POLITICA_DE_VERSION.get(version, "023")]) or caso["decision_motor"]


def huella_casos() -> str:
    """Huella corta de casos.json: si cambian los casos o sus referencias, cambia (issue #19)."""
    return hashlib.sha256((CARPETA / "casos.json").read_bytes()).hexdigest()[:10]


def propuesta(final: dict) -> str:
    """Lo que se compara con la política: desde la v4 la opinión del comité (la decisión final coincide con la
    política por construcción); antes, la decisión del comité."""
    if "opinion_comite" in final:
        return final["opinion_comite"] or "sin opinión"
    return final["decision"]


def borradores_con_violacion(historial: list) -> tuple[int, int]:
    """Issue #20: cuántos borradores revisó el auditor y cuántos violaban una regla dura."""
    auditados = [p["contenido"] for p in historial if p["nodo"] == "auditor"]
    return len(auditados), sum(bool(a.get("violaciones")) for a in auditados)
_lock = threading.Lock()


def _anexar(ruta, fila: dict) -> None:
    with _lock, open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps(limpio(fila), ensure_ascii=False) + "\n")


def _hechos(ruta) -> set:
    if not ruta.exists():
        return set()
    return {json.loads(l)["id"] for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()}


def datos_juez(historial: list, final: dict, solicitud: dict) -> dict:
    """Lo que ve el juez: lo mismo que vio el comité (solicitud, análisis y herramientas de los especialistas)."""
    especialistas = {p["nodo"]: p["contenido"] for p in historial if p["nodo"] in ("riesgo", "economia", "cupo")}
    return {"solicitud": solicitud,
            "especialistas": {n: {"analisis": e["analisis"], "herramientas": [h["salida"] for h in e["herramientas"]]}
                              for n, e in especialistas.items()},
            "decision": {k: final[k] for k in ("decision", "monto", "tasa")}
            | ({"opinion_comite": final["opinion_comite"]} if "opinion_comite" in final else {})}


def calificar(estado: dict, caso: dict, herr) -> dict:
    """Notas programadas (gratis) de una ejecución."""
    final, sol = estado["final"], a_df(estado["solicitud"])
    salidas = [h["salida"] for e in estado["especialistas"].values() for h in e["herramientas"]]
    oferta = None if final["decision"] in NO_OTORGA else herr.simular_oferta(sol, final["monto"], final["tasa"])
    extra = [oferta] if oferta else []
    if "decision_politica" in final:  # v4: la decisión de la política también es una fuente de cifras
        extra.append({k: final[k] for k in ("monto", "tasa")})
    fid, sin_respaldo = fidelidad(final.get("memo", ""), salidas + extra, resumen_solicitud(sol))
    sin_respaldo = {"numeros": sin_respaldo, "piso": piso_fidelidad(salidas + extra, resumen_solicitud(sol))}
    return {
        "concordancia": float(propuesta(final) == referencia(caso, VERSION)),
        "reglas_ok": float(not reglas_duras(final, oferta)),
        "fidelidad": fid,
        "auditoria_1er_intento": float(final["auditoria"]["aprobado"] and final["intentos"] == 1),
    }, sin_respaldo


def correr_caso(caso: dict, s2: pd.DataFrame, herr, cliente, modelo: str, carpeta) -> dict:
    sol = s2.iloc[[caso["posicion"]]]
    medidor = Medidor(cliente, modelo)
    inicio = time.perf_counter()
    estado = evaluar_solicitud(construir(medidor, herr, modelo, firma_humana=False), sol,
                               {"eval": VERSION, "caso": caso["id"]})
    segundos = time.perf_counter() - inicio
    nota, sin_respaldo = calificar(estado, caso, herr)
    final = estado["final"]
    juez = juzgar(cliente, final.get("memo", ""), datos_juez(estado["historial"], final, resumen_solicitud(sol)))
    v = juez["veredicto"]
    nota["juez"] = float(np.mean([v.get(k, False) for k in RUBRICA])) if v else None
    with registro.conectar() as con:
        registro.guardar(con, estado, medidor, segundos, modelo=modelo)
    (carpeta / "traces").mkdir(parents=True, exist_ok=True)
    (carpeta / "traces" / f"{caso['id']}.json").write_text(
        json.dumps(limpio({"historial": estado["historial"], "final": final, "juez": juez}), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return {
        "id": caso["id"], "tags": [caso["decision_motor"], "malo" if caso["malo_real"] else "bueno"],
        "modelo": modelo, "version": VERSION, "decision_agente": propuesta(final), "decision_final": final["decision"],
        "decision_motor": referencia(caso, VERSION), "casos": huella_casos(),
        "borradores": borradores_con_violacion(estado["historial"]),
        "monto": final["monto"], "tasa": final["tasa"], "intentos": final["intentos"],
        "grade": nota, "juez_criterios": {k: v.get(k) for k in RUBRICA} if v else None,
        "sin_respaldo": sin_respaldo, "malo_real": caso["malo_real"],
        "roi_real": float(rendimiento(sol).iloc[0]),
        "latency_s": segundos, "llamadas": medidor.llamadas,
        "usage": {"input_tokens": medidor.tokens_entrada, "output_tokens": medidor.tokens_salida},
        "cost_usd": medidor.costo_usd(), "judge_model": juez["modelo"], "judge_usage": juez["usage"],
    }


def correr(limite: int | None = None, modelo: str = "claude-haiku-5-5", hilos: int = HILOS,
           ids: list[str] | None = None) -> None:
    carpeta = CARPETA / modelo / VERSION
    carpeta.mkdir(parents=True, exist_ok=True)
    resultados, errores = carpeta / "results.jsonl", carpeta / "errors.jsonl"
    casos = json.loads((CARPETA / "casos.json").read_text(encoding="utf-8"))
    pendientes = [c for c in casos if c["id"] not in _hechos(resultados) and (not ids or c["id"] in ids)][:limite]
    print(f"{len(pendientes)} casos por correr con {modelo} (juez {JUEZ})", flush=True)
    s2, herr, cliente = cargar_muestra(), Herramientas(), cliente_anthropic()
    with ThreadPoolExecutor(hilos) as ex:
        futuros = {ex.submit(correr_caso, c, s2, herr, cliente, modelo, carpeta): c for c in pendientes}
        for f in as_completed(futuros):
            c = futuros[f]
            try:
                fila = f.result()
                _anexar(resultados, fila)
                print(f"{c['id']}: {fila['decision_agente']} (política: {fila['decision_motor']}) · "
                      f"fidelidad {fila['grade']['fidelidad']:.0%} · juez {fila['grade']['juez']} · "
                      f"USD {fila['cost_usd']:.4f}", flush=True)
            except Exception as e:  # el caso queda pendiente para la próxima corrida
                _anexar(errores, {"id": c["id"], "tipo": type(e).__name__, "detalle": str(e)[:500]})
                print(f"{c['id']}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
    reporte(modelo)


def tabla_md(df: pd.DataFrame, formato: str | dict = "{}") -> str:
    """Tabla markdown sin dependencias (índice como primera columna). `formato` puede ir por columna."""
    def fmt(v, col):
        f = formato.get(col, "{}") if isinstance(formato, dict) else formato
        return f.format(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else str(v)
    filas = [f"| {df.index.name or ''} | " + " | ".join(map(str, df.columns)) + " |",
             "|" + "---|" * (len(df.columns) + 1)]
    filas += [f"| {i} | " + " | ".join(fmt(v, c) for v, c in zip(fila, df.columns)) + " |"
              for i, fila in zip(df.index, df.to_numpy(dtype=object))]
    return "\n".join(filas)


def recalificar(modelo: str = "claude-haiku-5-5", version: str = VERSION) -> None:
    """Recalcula fidelidad y concordancia desde las trazas guardadas (sin llamar a ningún modelo), p. ej. tras
    corregir el calificador o cambiar las referencias de casos.json (issue #19)."""
    carpeta = CARPETA / modelo / version
    ruta = carpeta / "results.jsonl"
    filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    casos = {c["id"]: c for c in json.loads((CARPETA / "casos.json").read_text(encoding="utf-8"))}
    for f in filas:
        t = json.loads((carpeta / "traces" / f"{f['id']}.json").read_text(encoding="utf-8"))
        salidas = [h["salida"] for p in t["historial"] if p["nodo"] in ("riesgo", "economia", "cupo")
                   for h in p["contenido"]["herramientas"]]
        ofertas = [p["contenido"].get("oferta") for p in t["historial"] if p["nodo"] == "auditor"]
        ofertas += [p["contenido"] for p in t["historial"] if p["nodo"] == "politica"]  # v4
        fid, sin = fidelidad(t["final"].get("memo", ""), salidas + [o for o in ofertas if o],
                             casos[f["id"]]["solicitud"])
        f["grade"]["fidelidad"] = fid
        f["sin_respaldo"] = {"numeros": sin, "piso": piso_fidelidad(salidas + [o for o in ofertas if o],
                                                                    casos[f["id"]]["solicitud"])}
        f["decision_motor"] = referencia(casos[f["id"]], version)
        f["grade"]["concordancia"] = float(f["decision_agente"] == f["decision_motor"])
        f["borradores"] = borradores_con_violacion(t["historial"])
        f["casos"] = huella_casos()
    ruta.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas), encoding="utf-8")
    reporte(modelo, version)


def rejuzgar(modelo: str = "claude-haiku-5-5", version: str = VERSION) -> None:
    """Vuelve a juzgar los memos guardados (solo llama al juez, no al agente), p. ej. tras corregir lo que ve el juez."""
    carpeta = CARPETA / modelo / version
    ruta = carpeta / "results.jsonl"
    filas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    casos = {c["id"]: c for c in json.loads((CARPETA / "casos.json").read_text(encoding="utf-8"))}
    cliente = cliente_anthropic()

    def uno(f):
        p = carpeta / "traces" / f"{f['id']}.json"
        t = json.loads(p.read_text(encoding="utf-8"))
        juez = juzgar(cliente, t["final"].get("memo", ""), datos_juez(t["historial"], t["final"], casos[f["id"]]["solicitud"]))
        t["juez"] = juez
        p.write_text(json.dumps(limpio(t), ensure_ascii=False, indent=1), encoding="utf-8")
        v = juez["veredicto"]
        f["grade"]["juez"] = float(np.mean([v.get(k, False) for k in RUBRICA])) if v else None
        f["juez_criterios"] = {k: v.get(k) for k in RUBRICA} if v else None
        f["judge_model"], f["judge_usage"] = juez["modelo"], juez["usage"]

    with ThreadPoolExecutor(HILOS) as ex:
        list(ex.map(uno, filas))
    ruta.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas), encoding="utf-8")
    reporte(modelo, version)


def _costo_juez(f) -> float:
    entrada, salida = PRECIOS[JUEZ]
    u = f["judge_usage"]
    return (u["input_tokens"] * entrada + u["output_tokens"] * salida) / 1e6


def reporte(modelo: str = "claude-haiku-5-5", version: str = VERSION) -> pd.DataFrame:
    """Resumen en consola y en evals/agente/<modelo>/<version>/reporte.md."""
    carpeta = CARPETA / modelo / version
    filas = [json.loads(l) for l in (carpeta / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.json_normalize(filas)
    n = len(df)
    notas = {k: df[f"grade.{k}"].mean() for k in ("concordancia", "reglas_ok", "fidelidad", "auditoria_1er_intento", "juez")}
    criterios = {k: df[f"juez_criterios.{k}"].mean() for k in RUBRICA if f"juez_criterios.{k}" in df}
    otorga_agente = ~df["decision_agente"].isin(NO_OTORGA)  # un caso escalado no otorga hasta que lo firme un analista
    otorga_motor = df["decision_motor"].ne("rechazar")
    juez_nulos = int(df["grade.juez"].isna().sum())  # issue #21: formatos inválidos del juez, contados aparte
    borr = [f.get("borradores") or [0, 0] for f in filas]
    auditados, con_violacion = sum(b[0] for b in borr), sum(b[1] for b in borr)
    escalados = int(df.get("decision_final", df["decision_agente"]).eq("escalar").sum())
    pisos = [f["sin_respaldo"]["piso"] for f in filas if isinstance(f.get("sin_respaldo"), dict)]
    piso = {k: np.mean([x[k] for x in pisos]) for k in ("porcentajes", "montos")} if pisos else None

    def negocio(m):
        return {"préstamos": int(m.sum()), "mora real": df.loc[m, "malo_real"].mean(), "ROI real": df.loc[m, "roi_real"].mean()}

    negocio_tabla = pd.DataFrame({"Agente": negocio(otorga_agente), "Motor": negocio(otorga_motor),
                                  "Lending Club (todos)": negocio(pd.Series(True, index=df.index))}).T
    confusion = pd.crosstab(df["decision_motor"], df["decision_agente"], rownames=["política"],
                            colnames=["comité" if version >= "v4" else "agente"])
    costo_agente, costo_juez = df["cost_usd"].sum(), sum(_costo_juez(f) for f in filas)
    errores = carpeta / "errors.jsonl"
    n_err = len(errores.read_text(encoding="utf-8").splitlines()) if errores.exists() else 0

    lineas = [f"# Eval del agente · {modelo} · {version}", "",
              f"{n} casos · {n_err} intentos con error · juez {JUEZ} · costo agente USD {costo_agente:.2f} + juez USD {costo_juez:.2f}",
              f"Latencia media {df['latency_s'].mean():.0f} s · {df['llamadas'].mean():.1f} llamadas por caso",
              f"Referencia: política {POLITICA_DE_VERSION.get(version, '023')} · casos.json {huella_casos()}", "",
              "## Notas (promedio por caso)", "", "| Métrica | Valor |", "|---|---|"]
    lineas += [f"| {k} | {v:.0%} |" for k, v in notas.items()]
    lineas += [f"| borradores del comité con violación de reglas duras | {con_violacion} de {auditados} |",
               f"| casos escalados por no pasar la auditoría | {escalados} |",
               f"| notas del juez inválidas (excluidas del promedio) | {juez_nulos} |"]
    if piso:
        lineas += [f"| piso de la fidelidad: % al azar que pasaría / montos al azar | "
                   f"{piso['porcentajes']:.0%} / {piso['montos']:.0%} |"]
    lineas += ["",
               "`reglas_ok` mide la decisión final, que cumple por construcción; la fila de borradores muestra si el "
               "comité las cumple. La auditoría la hace el mismo modelo que redacta: no es una revisión independiente.",
               "Desde la v4 la decisión la toma la política y `concordancia` mide la opinión del comité."]
    lineas += ["", "## Juez por criterio", "", "| Criterio | Cumple |", "|---|---|"]
    lineas += [f"| {k} | {v:.0%} |" for k, v in criterios.items()]
    lineas += ["", "## Propuesta frente a la política", "", tabla_md(confusion), "",
               "## Resultado real de lo que se otorga (aprobar o contraoferta)", "",
               tabla_md(negocio_tabla, {"préstamos": "{:,.0f}", "mora real": "{:.1%}", "ROI real": "{:.1%}"}), "",
               f"Con {n} casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.",
               "La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación."]
    texto = "\n".join(lineas) + "\n"
    (carpeta / "reporte.md").write_text(texto, encoding="utf-8")
    print(texto)
    return df

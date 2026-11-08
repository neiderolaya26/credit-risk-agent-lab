import json
from types import SimpleNamespace as NS

from credit_lab import registro
from credit_lab.agente import Auditoria, Medidor, evaluar_solicitud, firmar
from test_agente import CONTRAOFERTA, OK, SOLICITUD, ClienteFalso, HerramientasFalsas, _grafo, _opinion


class ClienteConUso(ClienteFalso):
    """Igual que el cliente falso, pero cada respuesta reporta 100 tokens de entrada y 20 de salida."""

    def _create(self, **kw):
        r = super()._create(**kw)
        r.usage = NS(input_tokens=100, output_tokens=20)
        return r

    def _parse(self, **kw):
        r = super()._parse(**kw)
        r.usage = NS(input_tokens=100, output_tokens=20)
        return r


SOL_REAL = SOLICITUD.assign(malo=[1], funded_amnt=[10_000.0], total_pymnt=[6_000.0], collection_recovery_fee=[0.0])


def _correr(opiniones, auditorias, politica=CONTRAOFERTA):
    medidor = Medidor(ClienteConUso(opiniones, auditorias))
    grafo = _grafo(medidor, HerramientasFalsas(politica))
    return grafo, evaluar_solicitud(grafo, SOL_REAL), medidor


def test_medidor_suma_llamadas_tokens_y_costo():
    _, _, m = _correr([_opinion()], [OK])
    assert m.llamadas == 8  # 3 especialistas × 2 + comité + auditor
    assert (m.tokens_entrada, m.tokens_salida) == (800, 160)
    assert abs(m.costo_usd() - (800 * 2 + 160 * 10) / 1e6) < 1e-12


def test_guardar_registra_ejecucion_pasos_y_resultado_real(tmp_path):
    _, estado, m = _correr([_opinion("aprobar"), _opinion("contraoferta")],
                           [Auditoria(aprobado=False, hallazgos=["x"]), OK])
    with registro.conectar(tmp_path / "e.db") as con:
        n = registro.guardar(con, estado, m, segundos=1.5, semilla=3)
        e, p = registro.leer(con).iloc[0], registro.leer(con, "pasos")
    assert n == 1 and e["decision"] == "contraoferta" and e["intentos"] == 2
    assert e["decision_politica"] == "contraoferta" and e["opinion_comite"] == "contraoferta" and e["desacuerdo"] == 0
    assert e["malo_real"] == 1 and abs(e["roi_real"] - (-0.4)) < 1e-9
    assert e["estado"] == "pendiente de firma" and e["decision_final"] is None and e["hilo"] == estado["hilo"]
    assert sorted(p["nodo"].value_counts().items()) == [
        ("auditor", 2), ("comite", 2), ("cupo", 1), ("economia", 1), ("politica", 1), ("riesgo", 1)]
    assert json.loads(p.loc[p["nodo"].eq("auditor"), "contenido"].iloc[0])["aprobado"] is False


def test_resumen_solo_cuenta_decisiones_efectivas(tmp_path):
    """Issue #18: una propuesta devuelta por el analista no entra en la mora ni el ROI de su decisión."""
    with registro.conectar(tmp_path / "e.db") as con:
        for confirma in (True, False):
            grafo, estado, m = _correr([_opinion("contraoferta")], [OK])
            registro.guardar(con, estado, m)
            final = firmar(grafo, estado["hilo"], confirma, analista="ana",
                           motivo=None if confirma else "datos_inconsistentes")["final"]
            registro.actualizar_firma(con, estado["hilo"], final)
        _, aprobada, m = _correr([_opinion("aprobar")], [OK], politica={**CONTRAOFERTA, "decision": "aprobar"})
        registro.guardar(con, aprobada, m)
        res, k = registro.resumen(con), registro.kpis(con)
    assert res["ejecuciones"].sum() == 2  # la devuelta queda fuera
    assert res.loc[("contraoferta", "firmada por analista"), "ejecuciones"] == 1
    assert abs(k["aprobadas_automaticamente"] - 1 / 3) < 1e-9 and k["devueltas_sobre_firmadas"] == 0.5
    assert k["motivos_analista"] == {"confirma_politica": 1, "datos_inconsistentes": 1}

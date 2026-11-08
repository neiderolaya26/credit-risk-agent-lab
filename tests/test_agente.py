from types import SimpleNamespace as NS

import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from credit_lab.agente import (Auditoria, Opinion, a_df, a_dict, construir, evaluar_solicitud, firmar, reglas_duras,
                               validar_firma, validar_oferta)

SOLICITUD = pd.DataFrame({"loan_amnt": [10_000.0], "int_rate": [12.0], "annual_inc": [60_000.0],
                          "purpose": ["debt_consolidation"], "dti": [15.0], "fico_range_low": [700.0],
                          "issue_d": [pd.Timestamp("2015-08-01")]})
OFERTA_OK = {"cuota_ingreso": 0.07, "roi_esperado": 0.05}
APROBAR = {"decision": "aprobar", "regla": "1", "monto": 10_000.0, "tasa": 12.0}
CONTRAOFERTA = {"decision": "contraoferta", "regla": "2b", "monto": 8_000.0, "tasa": 12.0}
OK = Auditoria(aprobado=True, hallazgos=[])


class HerramientasFalsas:
    def __init__(self, politica=APROBAR):
        self._politica = politica

    def politica(self, sol):
        return dict(self._politica)

    def razones_cliente(self, sol):
        return [{"variable": "dti", "texto": "Su nivel de endeudamiento frente a su ingreso es alto."}]

    def riesgo(self, sol):
        return {"pd": 0.12}

    def economia(self, sol):
        return {"roi_esperado": 0.05}

    def cupo_pricing(self, sol):
        return {"monto_maximo": 20_000, "tasa_sugerida": 9.5}

    def simular_oferta(self, sol, monto, tasa):
        return {"monto": monto, "tasa": tasa, **OFERTA_OK}


class ClienteFalso:
    """Especialistas: piden su primera herramienta y luego responden texto. Comité y auditor: respuestas fijas."""

    def __init__(self, opiniones, auditorias, siempre_herramienta=False):
        self.opiniones, self.auditorias = list(opiniones), list(auditorias)
        self.herramientas_usadas, self.siempre_herramienta = [], siempre_herramienta
        self.beta = NS(messages=NS(create=self._create, parse=self._parse))

    def _create(self, **kw):
        ultimo = kw["messages"][-1]["content"]
        if isinstance(ultimo, list) and not self.siempre_herramienta:  # ya llegaron los tool_result
            return NS(stop_reason="end_turn", content=[NS(type="text", text="señal: favorable")])
        nombre = kw["tools"][0]["name"]
        self.herramientas_usadas.append(nombre)
        return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=f"t_{nombre}", name=nombre, input={})])

    def _parse(self, **kw):
        fuente = self.opiniones if kw["output_format"] is Opinion else self.auditorias
        return NS(stop_reason="end_turn", parsed_output=fuente.pop(0))


def _opinion(tipo="aprobar"):
    return Opinion(opinion=tipo, razones=["r"], que_verificar=[], memo="memo")


def _grafo(cliente, herr=None, **kw):
    # serializador estricto (sin pickle): prueba que el estado se puede guardar sin objetos arbitrarios (issue #26)
    kw.setdefault("checkpointer", InMemorySaver(serde=JsonPlusSerializer()))
    return construir(cliente, herr or HerramientasFalsas(), **kw)


def test_reglas_duras():
    assert reglas_duras({"decision": "rechazar", "monto": 0, "tasa": 0}, None) == []
    assert reglas_duras({"decision": "escalar", "monto": 50_000, "tasa": 3}, None) == []
    assert reglas_duras({"decision": "aprobar", "monto": 10_000, "tasa": 12}, OFERTA_OK) == []
    v = reglas_duras({"decision": "aprobar", "monto": 50_000, "tasa": 3},
                     {"cuota_ingreso": 0.35, "roi_esperado": -0.01})
    assert len(v) == 4


def test_aprobacion_con_doble_llave_sale_sin_firma():
    cliente = ClienteFalso([_opinion()], [OK])
    estado = evaluar_solicitud(_grafo(cliente), SOLICITUD)
    assert set(estado["especialistas"]) == {"riesgo", "economia", "cupo"}
    assert sorted(cliente.herramientas_usadas) == ["consultar_cupo_pricing", "consultar_economia", "consultar_riesgo"]
    f = estado["final"]
    assert "__interrupt__" not in estado and f["estado"] == "aprobada automáticamente"
    assert f["decision"] == f["decision_final"] == "aprobar" and not f["desacuerdo"]


def test_la_decision_es_de_la_politica_aunque_el_comite_diga_aprobar():
    """Issue #17: un 'aprobar' del LLM no puede saltarse la firma ni cambiar las condiciones."""
    estado = evaluar_solicitud(_grafo(ClienteFalso([_opinion("aprobar")], [OK]), HerramientasFalsas(CONTRAOFERTA)),
                               SOLICITUD)
    f = estado["final"]
    assert (f["decision"], f["monto"]) == ("contraoferta", 8_000.0) and f["desacuerdo"]
    assert f["estado"] == "pendiente de firma" and estado["__interrupt__"][0].value["desacuerdo"] is True


def test_desacuerdo_del_comite_con_una_aprobacion_va_a_firma():
    estado = evaluar_solicitud(_grafo(ClienteFalso([_opinion("rechazar")], [OK])), SOLICITUD)
    f = estado["final"]
    assert f["decision"] == "aprobar" and f["opinion_comite"] == "rechazar" and f["estado"] == "pendiente de firma"
    assert f["decision_final"] is None


def test_auditor_devuelve_al_comite_y_luego_aprueba():
    cliente = ClienteFalso([_opinion(), _opinion()], [Auditoria(aprobado=False, hallazgos=["número no coincide"]), OK])
    f = evaluar_solicitud(_grafo(cliente), SOLICITUD)["final"]
    assert f["intentos"] == 2 and f["estado"] == "aprobada automáticamente"


def test_auditoria_fallida_escala_con_la_propuesta_de_la_politica():
    malo = Auditoria(aprobado=False, hallazgos=["dato inventado"])
    f = evaluar_solicitud(_grafo(ClienteFalso([_opinion()] * 2, [malo] * 2)), SOLICITUD)["final"]
    assert f["decision"] == "escalar" and f["propuesta_original"]["decision"] == "aprobar"  # no es un rechazo (022)
    assert f["estado"] == "pendiente de firma"


def test_contraoferta_trae_razones_y_carta_para_el_cliente():
    f = evaluar_solicitud(_grafo(ClienteFalso([_opinion("contraoferta")], [OK]), HerramientasFalsas(CONTRAOFERTA)),
                          SOLICITUD)["final"]
    assert f["razones_cliente"][0]["variable"] == "dti"
    assert "USD 8,000" in f["carta_cliente"] and "endeudamiento" in f["carta_cliente"]


def test_firma_confirmada_y_devuelta():
    """Issue #18: lo devuelto por el analista no es una decisión."""
    for confirma, esperado in [(True, "contraoferta"), (False, None)]:
        grafo = _grafo(ClienteFalso([_opinion("contraoferta")], [OK]), HerramientasFalsas(CONTRAOFERTA))
        estado = evaluar_solicitud(grafo, SOLICITUD)
        estado = firmar(grafo, estado["hilo"], confirma, analista="ana", motivo=None if confirma else "capacidad_pago")
        assert estado["final"]["decision_final"] == esperado
        assert estado["final"]["firma"]["motivo"] == ("confirma_politica" if confirma else "capacidad_pago")


def test_dos_solicitudes_en_el_mismo_grafo_no_comparten_estado():
    """Issue #25: cada solicitud tiene su propio hilo."""
    grafo = _grafo(ClienteFalso([_opinion()] * 2, [OK] * 2))
    a, b = evaluar_solicitud(grafo, SOLICITUD), evaluar_solicitud(grafo, SOLICITUD)
    assert a["hilo"] != b["hilo"] and b["intentos"] == 1
    assert sum(p["nodo"] == "comite" for p in b["historial"]) == 1


def test_la_firma_sobrevive_al_proceso(tmp_path):
    """Issue #26: con el checkpointer SQLite, otro grafo (otro proceso) retoma la propuesta pendiente."""
    from credit_lab.agente import checkpointer_sqlite

    ruta = tmp_path / "firmas.db"
    primero = construir(ClienteFalso([_opinion("contraoferta")], [OK]), HerramientasFalsas(CONTRAOFERTA),
                        checkpointer=checkpointer_sqlite(ruta))
    hilo = evaluar_solicitud(primero, SOLICITUD)["hilo"]
    segundo = construir(ClienteFalso([], []), HerramientasFalsas(CONTRAOFERTA), checkpointer=checkpointer_sqlite(ruta))
    estado = firmar(segundo, hilo, True, analista="ana")
    assert estado["final"]["estado"] == "firmada por analista"


def test_validar_firma_rechaza_respuestas_mal_formadas():
    for mala in [None, "sí", {"confirma": "sí", "analista": "a"}, {"confirma": True},
                 {"confirma": False, "analista": "a"}, {"confirma": True, "analista": "a", "motivo": "porque sí"}]:
        with pytest.raises(ValueError):
            validar_firma(mala)


def test_validar_oferta():
    assert validar_oferta(10_000, 12) == (10_000.0, 12.0)
    for monto, tasa in [(-5, 12), (1e9, 12), (10_000, 0), (10_000, 45), ("x", 12), (float("nan"), 12)]:
        with pytest.raises(ValueError):
            validar_oferta(monto, tasa)


def test_especialista_que_agota_los_turnos_queda_marcado():
    """Issue #31."""
    estado = evaluar_solicitud(_grafo(ClienteFalso([_opinion()], [OK], siempre_herramienta=True), firma_humana=False),
                               SOLICITUD)
    assert all(e["agotado"] for e in estado["especialistas"].values())
    assert "incompleto" in estado["especialistas"]["riesgo"]["analisis"]


def test_la_solicitud_viaja_como_dict_y_vuelve_igual():
    d = a_dict(SOLICITUD)
    assert d["issue_d"] == {"__fecha__": "2015-08-01T00:00:00"}
    pd.testing.assert_frame_equal(a_df(d), SOLICITUD, check_dtype=False)


def test_modo_eval_registra_la_propuesta_sin_firma():
    cliente = ClienteFalso([_opinion("contraoferta")], [OK])
    estado = evaluar_solicitud(construir(cliente, HerramientasFalsas(CONTRAOFERTA), firma_humana=False), SOLICITUD)
    assert "__interrupt__" not in estado and estado["final"]["estado"] == "propuesta sin firma (eval)"

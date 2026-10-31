from types import SimpleNamespace as NS

import pandas as pd

from credit_lab.agente import Auditoria, Decision, construir, evaluar_solicitud, reglas_duras

SOLICITUD = pd.DataFrame({"loan_amnt": [10_000.0], "int_rate": [12.0], "annual_inc": [60_000.0],
                          "purpose": ["debt_consolidation"], "dti": [15.0], "fico_range_low": [700.0]})
OFERTA_OK = {"cuota_ingreso": 0.07, "roi_esperado": 0.05}


class HerramientasFalsas:
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

    def __init__(self, decisiones, auditorias):
        self.decisiones, self.auditorias = list(decisiones), list(auditorias)
        self.herramientas_usadas = []
        self.beta = NS(messages=NS(create=self._create, parse=self._parse))

    def _create(self, **kw):
        ultimo = kw["messages"][-1]["content"]
        if isinstance(ultimo, list):  # ya llegaron los tool_result
            return NS(stop_reason="end_turn", content=[NS(type="text", text="señal: favorable")])
        nombre = kw["tools"][0]["name"]
        self.herramientas_usadas.append(nombre)
        return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=f"t_{nombre}", name=nombre, input={})])

    def _parse(self, **kw):
        fuente = self.decisiones if kw["output_format"] is Decision else self.auditorias
        return NS(stop_reason="end_turn", parsed_output=fuente.pop(0))


def _decision(tipo="aprobar", monto=10_000.0, tasa=12.0):
    return Decision(decision=tipo, monto=monto, tasa=tasa, razones=["r"], memo="memo")


def test_reglas_duras():
    assert reglas_duras({"decision": "rechazar", "monto": 0, "tasa": 0}, None) == []
    assert reglas_duras({"decision": "aprobar", "monto": 10_000, "tasa": 12}, OFERTA_OK) == []
    v = reglas_duras({"decision": "aprobar", "monto": 50_000, "tasa": 3},
                     {"cuota_ingreso": 0.35, "roi_esperado": -0.01})
    assert len(v) == 4


def test_flujo_aprobado_en_un_intento_con_especialistas_en_paralelo():
    cliente = ClienteFalso([_decision()], [Auditoria(aprobado=True, hallazgos=[])])
    estado = evaluar_solicitud(construir(cliente, HerramientasFalsas()), SOLICITUD)
    assert set(estado["especialistas"]) == {"riesgo", "economia", "cupo"}
    assert sorted(cliente.herramientas_usadas) == ["consultar_cupo_pricing", "consultar_economia", "consultar_riesgo"]
    assert estado["final"]["decision"] == "aprobar" and estado["final"]["intentos"] == 1


def test_auditor_devuelve_al_comite_y_luego_aprueba():
    cliente = ClienteFalso([_decision(), _decision("contraoferta", 8_000.0)],
                           [Auditoria(aprobado=False, hallazgos=["número no coincide"]),
                            Auditoria(aprobado=True, hallazgos=[])])
    final = evaluar_solicitud(construir(cliente, HerramientasFalsas()), SOLICITUD)["final"]
    assert final["decision"] == "contraoferta" and final["monto"] == 8_000.0 and final["intentos"] == 2


def test_violacion_de_regla_impide_aprobar_aunque_el_auditor_apruebe():
    cliente = ClienteFalso([_decision(tasa=45.0)] * 2, [Auditoria(aprobado=True, hallazgos=[])] * 2)
    final = evaluar_solicitud(construir(cliente, HerramientasFalsas()), SOLICITUD)["final"]
    assert final["decision"] == "rechazar" and final["intentos"] == 2
    assert any("tasa fuera de la grilla" in h for h in final["auditoria"]["hallazgos"])


def test_aprobacion_no_espera_firma():
    cliente = ClienteFalso([_decision()], [Auditoria(aprobado=True, hallazgos=[])])
    estado = evaluar_solicitud(construir(cliente, HerramientasFalsas()), SOLICITUD, hilo="a")
    assert "__interrupt__" not in estado and estado["final"]["estado"] == "aprobada automáticamente"


def test_contraoferta_espera_firma_y_el_analista_la_confirma():
    from credit_lab.agente import firmar

    cliente = ClienteFalso([_decision("contraoferta", 8_000.0)], [Auditoria(aprobado=True, hallazgos=[])])
    grafo = construir(cliente, HerramientasFalsas())
    estado = evaluar_solicitud(grafo, SOLICITUD, hilo="b")
    assert estado["final"]["estado"] == "pendiente de firma"
    assert estado["__interrupt__"][0].value["propuesta"]["monto"] == 8_000.0
    estado = firmar(grafo, confirma=True, analista="ana", comentario="ok", hilo="b")
    assert estado["final"]["estado"] == "firmada por analista" and estado["final"]["firma"]["analista"] == "ana"


def test_modo_eval_registra_la_propuesta_sin_firma():
    cliente = ClienteFalso([_decision("contraoferta", 8_000.0)], [Auditoria(aprobado=True, hallazgos=[])])
    estado = evaluar_solicitud(construir(cliente, HerramientasFalsas(), firma_humana=False), SOLICITUD)
    assert "__interrupt__" not in estado and estado["final"]["estado"] == "propuesta sin firma (eval)"

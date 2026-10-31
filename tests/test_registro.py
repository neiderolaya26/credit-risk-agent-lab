import json
from types import SimpleNamespace as NS

from credit_lab import registro
from credit_lab.agente import Auditoria, Medidor, construir, evaluar_solicitud
from test_agente import SOLICITUD, ClienteFalso, HerramientasFalsas, _decision


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


def _correr(decisiones, auditorias):
    medidor = Medidor(ClienteConUso(decisiones, auditorias))
    sol = SOLICITUD.assign(malo=[1], funded_amnt=[10_000.0], total_pymnt=[6_000.0], collection_recovery_fee=[0.0])
    return evaluar_solicitud(construir(medidor, HerramientasFalsas()), sol), medidor


def test_medidor_suma_llamadas_tokens_y_costo():
    _, m = _correr([_decision()], [Auditoria(aprobado=True, hallazgos=[])])
    assert m.llamadas == 8  # 3 especialistas × 2 + comité + auditor
    assert (m.tokens_entrada, m.tokens_salida) == (800, 160)
    assert abs(m.costo_usd() - (800 * 2 + 160 * 10) / 1e6) < 1e-12


def test_guardar_registra_ejecucion_pasos_y_resultado_real(tmp_path):
    estado, m = _correr([_decision(), _decision("contraoferta", 8_000.0)],
                        [Auditoria(aprobado=False, hallazgos=["x"]), Auditoria(aprobado=True, hallazgos=[])])
    with registro.conectar(tmp_path / "e.db") as con:
        n = registro.guardar(con, estado, m, segundos=1.5, semilla=3)
        e, p = registro.leer(con).iloc[0], registro.leer(con, "pasos")
        res = registro.resumen(con)
    assert n == 1 and e["decision"] == "contraoferta" and e["intentos"] == 2
    assert e["malo_real"] == 1 and abs(e["roi_real"] - (-0.4)) < 1e-9
    assert sorted(p["nodo"].value_counts().items()) == [("auditor", 2), ("comite", 2), ("cupo", 1), ("economia", 1), ("riesgo", 1)]
    assert json.loads(p.loc[p["nodo"].eq("auditor"), "contenido"].iloc[0])["aprobado"] is False
    assert res.loc["contraoferta", "ejecuciones"] == 1

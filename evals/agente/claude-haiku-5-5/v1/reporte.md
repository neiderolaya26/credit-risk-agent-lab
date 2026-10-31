# Eval del agente · claude-haiku-5-5 · v1

5 casos · 0 intentos con error · juez claude-opus-5-5 · costo agente USD 0.04 + juez USD 0.18
Latencia media 60 s · 9.4 llamadas por caso

## Notas (promedio por caso)

| Métrica | Valor |
|---|---|
| concordancia | 100% |
| reglas_ok | 100% |
| fidelidad | 93% |
| auditoria_1er_intento | 80% |
| juez | 76% |

## Juez por criterio

| Criterio | Cumple |
|---|---|
| justifica_con_numeros | 100% |
| coherente | 100% |
| alertas | 80% |
| sin_invenciones | 20% |
| claro | 80% |

## Decisión del agente frente a la política del motor

| motor | aprobar | contraoferta | rechazar |
|---|---|---|---|
| aprobar | 2 | 0 | 0 |
| contraoferta | 0 | 2 | 0 |
| rechazar | 0 | 0 | 1 |

## Resultado real de lo que se otorga (aprobar o contraoferta)

|  | préstamos | mora real | ROI real |
|---|---|---|---|
| Agente | 4 | 50.0% | 4.7% |
| Motor | 4 | 50.0% | 4.7% |
| Lending Club (todos) | 5 | 60.0% | -9.6% |

Con 30 casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.
La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación.

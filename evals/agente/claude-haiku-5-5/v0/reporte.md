# Eval del agente · claude-haiku-5-5 · v0

5 casos · 0 intentos con error · juez claude-opus-5-5 · costo agente USD 0.06 + juez USD 0.18
Latencia media 84 s · 11.0 llamadas por caso

## Notas (promedio por caso)

| Métrica | Valor |
|---|---|
| concordancia | 20% |
| reglas_ok | 100% |
| fidelidad | 93% |
| auditoria_1er_intento | 0% |
| juez | 56% |

## Juez por criterio

| Criterio | Cumple |
|---|---|
| justifica_con_numeros | 100% |
| coherente | 20% |
| alertas | 100% |
| sin_invenciones | 0% |
| claro | 60% |

## Decisión del agente frente a la política del motor

| motor | rechazar |
|---|---|
| aprobar | 2 |
| contraoferta | 2 |
| rechazar | 1 |

## Resultado real de lo que se otorga (aprobar o contraoferta)

|  | préstamos | mora real | ROI real |
|---|---|---|---|
| Agente | 0.000 | nan | nan |
| Motor | 4.000 | 0.500 | 0.047 |
| Lending Club (todos) | 5.000 | 0.600 | -0.096 |

Con 30 casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.
La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación.

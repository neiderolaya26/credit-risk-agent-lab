# Eval del agente · claude-haiku-5-5 · v2

30 casos · 0 intentos con error · juez claude-opus-5-5 · costo agente USD 0.22 + juez USD 1.24
Latencia media 51 s · 9.0 llamadas por caso

## Notas (promedio por caso)

| Métrica | Valor |
|---|---|
| concordancia | 83% |
| reglas_ok | 100% |
| fidelidad | 94% |
| auditoria_1er_intento | 93% |
| juez | 97% |

## Juez por criterio

| Criterio | Cumple |
|---|---|
| justifica_con_numeros | 100% |
| coherente | 100% |
| alertas | 87% |
| sin_invenciones | 97% |
| claro | 100% |

## Decisión del agente frente a la política del motor

| motor | aprobar | contraoferta | rechazar |
|---|---|---|---|
| aprobar | 16 | 0 | 0 |
| contraoferta | 0 | 8 | 5 |
| rechazar | 0 | 0 | 1 |

## Resultado real de lo que se otorga (aprobar o contraoferta)

|  | préstamos | mora real | ROI real |
|---|---|---|---|
| Agente | 24.000 | 0.375 | -0.051 |
| Motor | 29.000 | 0.379 | -0.023 |
| Lending Club (todos) | 30.000 | 0.400 | -0.045 |

Con 30 casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.
La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación.

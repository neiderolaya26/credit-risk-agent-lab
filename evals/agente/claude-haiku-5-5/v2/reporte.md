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
| Agente | 24 | 37.5% | -5.1% |
| Motor | 29 | 37.9% | -2.3% |
| Lending Club (todos) | 30 | 40.0% | -4.5% |

Con 30 casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.
La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación.

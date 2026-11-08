# Eval del agente · claude-haiku-5-5 · v3

35 casos · 0 intentos con error · juez claude-opus-5-5 · costo agente USD 0.24 + juez USD 1.43
Latencia media 46 s · 8.5 llamadas por caso
Referencia: política 020 · casos.json 67a8d0abe5

## Notas (promedio por caso)

| Métrica | Valor |
|---|---|
| concordancia | 100% |
| reglas_ok | 100% |
| fidelidad | 94% |
| auditoria_1er_intento | 100% |
| juez | 98% |
| borradores del comité con violación de reglas duras | 0 de 35 |
| casos escalados por no pasar la auditoría | 0 |
| notas del juez inválidas (excluidas del promedio) | 0 |
| piso de la fidelidad: % al azar que pasaría / montos al azar | 13% / 3% |

`reglas_ok` mide la decisión final, que cumple por construcción; la fila de borradores muestra si el comité las cumple. La auditoría la hace el mismo modelo que redacta: no es una revisión independiente.
Desde la v4 la decisión la toma la política y `concordancia` mide la opinión del comité.

## Juez por criterio

| Criterio | Cumple |
|---|---|
| justifica_con_numeros | 100% |
| coherente | 100% |
| alertas | 97% |
| sin_invenciones | 94% |
| claro | 100% |

## Propuesta frente a la política

| política | aprobar | contraoferta | rechazar |
|---|---|---|---|
| aprobar | 16 | 0 | 0 |
| contraoferta | 0 | 14 | 0 |
| rechazar | 0 | 0 | 5 |

## Resultado real de lo que se otorga (aprobar o contraoferta)

|  | préstamos | mora real | ROI real |
|---|---|---|---|
| Agente | 30 | 40.0% | -4.5% |
| Motor | 30 | 40.0% | -4.5% |
| Lending Club (todos) | 35 | 42.9% | -5.9% |

Con 35 casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.
La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación.

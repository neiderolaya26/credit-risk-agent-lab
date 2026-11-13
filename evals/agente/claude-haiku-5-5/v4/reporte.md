# Eval del agente · claude-haiku-5-5 · v4

35 casos · 9 intentos con error · juez claude-opus-5-5 · costo agente USD 0.29 + juez USD 0.00
Latencia media 53 s · 8.7 llamadas por caso
Referencia: política 023 · casos.json 6f4d3976ad

## Notas (promedio por caso)

| Métrica | Valor |
|---|---|
| concordancia | 97% |
| reglas_ok | 100% |
| fidelidad | 94% |
| auditoria_1er_intento | 94% |
| juez | sin dato |
| borradores del comité con violación de reglas duras | 0 de 37 |
| casos escalados por no pasar la auditoría | 0 |
| notas del juez inválidas (excluidas del promedio) | 0 |
| casos sin juez (se corre después con `rejuzgar`) | 35 |
| piso de la fidelidad: % al azar que pasaría / montos al azar | 13% / 3% |

`reglas_ok` mide la decisión final, que cumple por construcción; la fila de borradores muestra si el comité las cumple. La auditoría la hace el mismo modelo que redacta: no es una revisión independiente.
Desde la v4 la decisión la toma la política y `concordancia` mide la opinión del comité.

## Juez por criterio

| Criterio | Cumple |
|---|---|

## Propuesta frente a la política

| política | aprobar | contraoferta | rechazar |
|---|---|---|---|
| aprobar | 15 | 1 | 0 |
| contraoferta | 0 | 13 | 0 |
| rechazar | 0 | 0 | 6 |

## Resultado real de lo que se otorga (aprobar o contraoferta)

|  | préstamos | mora real | ROI real |
|---|---|---|---|
| Agente | 29 | 37.9% | -3.4% |
| Motor | 29 | 37.9% | -3.4% |
| Lending Club (todos) | 35 | 42.9% | -5.9% |

Con 35 casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.
La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación.

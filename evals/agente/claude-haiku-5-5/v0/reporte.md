# Eval del agente · claude-haiku-5-5 · v0

5 casos · 0 intentos con error · juez claude-opus-5-5 · costo agente USD 0.06 + juez USD 0.18
Latencia media 84 s · 11.0 llamadas por caso
Referencia: política 020 · casos.json 6f4d3976ad

## Notas (promedio por caso)

| Métrica | Valor |
|---|---|
| concordancia | 0% |
| reglas_ok | 100% |
| fidelidad | 93% |
| auditoria_1er_intento | 0% |
| juez | 56% |
| borradores del comité con violación de reglas duras | 0 de 10 |
| casos escalados por no pasar la auditoría | 0 |
| notas del juez inválidas (excluidas del promedio) | 0 |
| piso de la fidelidad: % al azar que pasaría / montos al azar | 12% / 3% |

`reglas_ok` mide la decisión final, que cumple por construcción; la fila de borradores muestra si el comité las cumple. La auditoría la hace el mismo modelo que redacta: no es una revisión independiente.
Desde la v4 la decisión la toma la política y `concordancia` mide la opinión del comité.

## Juez por criterio

| Criterio | Cumple |
|---|---|
| justifica_con_numeros | 100% |
| coherente | 20% |
| alertas | 100% |
| sin_invenciones | 0% |
| claro | 60% |

## Propuesta frente a la política

| política | rechazar |
|---|---|
| aprobar | 2 |
| contraoferta | 3 |

## Resultado real de lo que se otorga (aprobar o contraoferta)

|  | préstamos | mora real | ROI real |
|---|---|---|---|
| Agente | 0 | nan% | nan% |
| Motor | 5 | 60.0% | -9.6% |
| Lending Club (todos) | 5 | 60.0% | -9.6% |

Con 5 casos estas diferencias son ruidosas: sirven para detectar errores grandes, no para declarar un ganador.
La contraoferta cambia monto o tasa; se usa el resultado real del préstamo original como aproximación.

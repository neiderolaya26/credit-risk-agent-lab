# Fichas de los modelos y del agente

Documentación corta, al estilo de un inventario de modelos (SR 11-7): qué hace cada componente que decide o informa una decisión, con qué datos, cómo rinde, qué no hace y cómo se vigila. Issue #40.

## Modelo de riesgo (campeón)

| Campo | Contenido |
|---|---|
| Qué hace | Estima la PD (probabilidad de mora) de una solicitud a 36 meses |
| Algoritmo | LightGBM recalibrado en el intercepto (decisiones 006 y 007) |
| Datos | Lending Club, préstamos a 36 meses de sep-2012 a 2014 (entrenamiento); 2015 fuera de tiempo. Solo variables de la solicitud; sin la tasa ni el score de Lending Club |
| Desempeño (2015) | AUC 0,6823 frente a 0,6784 del score de Lending Club; IC 95% de la diferencia [+0,0016; +0,0063] |
| Calibración | Desplazamiento ajustado con ene–jun 2015; medida en jul–dic 2015: PD media frente a mora real dentro de ±0,7 pts en cada grupo de vivienda y región |
| Equidad | Sin atributos protegidos en los datos. La antigüedad crediticia (proxy de edad) no se usa como razón; quitarla del modelo cuesta −0,0032 de AUC y forzarla monótona −0,0011 (`python -m credit_lab.proxies`) |
| Límites | Solo préstamos que Lending Club aprobó: sin solicitudes rechazadas ni datos de aceptación |
| Monitoreo propuesto | PSI de la PD y de las 5 variables principales (alerta > 0,10, acción > 0,25); calibración mensual por grupo; AUC con 12 meses de maduración |

## Prepago y pérdida

| Campo | Contenido |
|---|---|
| Qué hacen | Rendimiento si el préstamo sale bueno (prepago por supervivencia en tiempo discreto) y si sale malo (pérdida económica) |
| Desempeño | Supervivencia: RMSE 0,0497 (decisión 011). La LGD clásica no tuvo señal |
| Límites | El efecto causal de la tasa en la mora no se pudo identificar (decisión 015): la PD es fija ante la tasa y el estrés es un supuesto |

## Política de crédito (código, sin LLM)

| Campo | Contenido |
|---|---|
| Qué hace | Decide aprobar, contraofertar o rechazar, con la oferta (`motor.politica_lote`, decisión 023) |
| Reglas | Aprobar con ROI esperado ≥ 4% y cuota ≤ 20% del ingreso; si no, contraoferta bajando el monto y después subiendo la tasa (máximo +2 pts y dentro de la banda de mercado); si no, rechazar |
| Backtest jul–dic 2015 (10.000 solicitudes) | 84,9% aprueba, 13,8% contraoferta, 1,3% rechaza; 15,1% iría a firma (126–202 horas de analista por cada 10.000 solicitudes a 5–8 minutos por caso) |
| Supuestos | Costos de fondeo, operación y capital y aceptación de contraofertas: no hay datos; el P&L usa valores dummy ilustrativos (ver `credit_lab.valor`) |

## Agente (LLM)

| Versión | Qué cambió | Eval (Haiku 5.5, juez Opus 5.5) | Referencia |
|---|---|---|---|
| v0 | Piloto | 5 casos · concordancia 0% | política 020 |
| v1 | Política compartida, auditor por materialidad | 5 casos · 80% | política 020 |
| v2 | Memo sin invenciones | 30 casos · 80% · juez 97% | política 020 |
| v3 | Regla 2b con ROI ≥ 4% (decisión 020) | 35 casos · 100% · juez 98% | política 020 |
| **v4** | El código decide; el comité opina y redacta (decisión 023) | 35 casos · opinión del comité 97% (decisión final 100% por construcción) · juez pendiente | política 023 |

- **Qué hace:** tres especialistas analizan con herramientas, el comité da su opinión y redacta el memo, un auditor lo revisa. La decisión y las razones para el cliente salen del código.
- **Control humano:** se aprueba sola solo si la política aprueba, el comité coincide y la auditoría pasa; lo demás lo firma un analista con un motivo de una lista cerrada (decisiones 021–023).
- **Regla de uso:** el CI falla si la versión por defecto no tiene su eval con los casos vigentes y concordancia ≥ 80% (`python -m credit_lab.evals verificar`).
- **Límites de la eval:** 35 casos detectan errores grandes, no diferencias finas; el juez no está calibrado con etiquetas humanas (issue #21); la auditoría la hace el mismo modelo que redacta, así que no es independiente; con las fuentes de un caso, un porcentaje al azar pasaría la métrica de fidelidad 12% de las veces (piso, issue #23).
- **Monitoreo propuesto:** % aprobado solo, % a firma, % devuelto por el analista y sus motivos, desacuerdos comité–política y minutos hasta la firma (`registro.kpis`).

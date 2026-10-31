# Registro de decisiones

Cada decisión importante del proyecto queda aquí, **escrita por Neider**: qué se decidió, qué alternativas había y por qué.

---

## 001 — Población, variable objetivo y benchmark

- **Fecha:** 2026-10-30
- **Contexto:** Lending Club (EE. UU.) tiene 2,26 M de préstamos aprobados de 2007 a 2018, con pagos observados hasta mar-2019. El 39% sigue vigente y los años recientes no han madurado: entre los préstamos terminados, la tasa de castigo de 2016–2017 a 36 meses sale ~20% frente a ~14% en los años maduros (sesgo de censura). Ver `notebooks/01_eda_poblacion.ipynb`.
- **Decisión:**
  - **Población:** préstamos a **36 meses** originados entre **2012 y 2015** (~589 mil préstamos), excluyendo los marcados "Does not meet the credit policy".
  - **Variable objetivo:** malo = *Charged Off* o *Default*; bueno = *Fully Paid*.
  - **Benchmark:** `int_rate` (y `grade`/`sub_grade`) **no se usan como features**. Se usan como el **modelo campeón** de Lending Club para medir contra él nuestro modelo **retador (challenger)**.
  - **Validación:** entrenar con 2012–2014 y validar fuera de tiempo con 2015.
- **Alternativas consideradas:** todos los préstamos terminados de 2007–2018 (descartada por el sesgo de censura); incluir los de 60 meses (otro perfil de riesgo: 25% de malos y solo ~55 mil maduros); empezar en 2007 (sin `mort_acc` ni `bc_util` antes de 2012); contar la mora de 31–120 días como malo (casi no existe en los años maduros).
- **Por qué (en mis palabras):** solo 36 meses ya que se evalúan los madurados; 2012–2015 para no tener un training tan malo; `int_rate` como benchmark para medir el modelo challenger.

---

## 002 — Selección de variables por familia de modelo

- **Fecha:** 2026-10-30
- **Contexto:** con el entrenamiento 2012–2014, la mejor variable propia (FICO) tiene un IV de 0,13, frente a 0,29–0,31 del score de Lending Club. La mora previa casi no predice (IV ≈ 0) por el sesgo de selección de la aprobación. Ver `notebooks/02_eda_variables.ipynb`.
- **Decisión:**
  - **Regresión logística:** 16 variables limpias con IV ≥ 0,02.
  - **Árboles (RF, XGBoost, LightGBM) y red neuronal:** 31 variables limpias, incluidas las de IV bajo, porque pueden servir en interacciones.
  - **Excluidas:** `installment` (se calcula con `int_rate`), `funded_amnt`, `fico_range_low/high` (se usa `fico_medio`), `initial_list_status`, `application_type` (constante) y **`verification_status`**.
  - Listas en `src/credit_lab/features.py`.
- **Alternativas consideradas:** A) solo IV ≥ 0,02 para todos los modelos; B) todas las variables limpias para todos los modelos.
- **Por qué (en mis palabras):** C para poder evaluar todos los modelos, y `verification_status` la excluyo porque es política de LC.

---

## 003 — Inicio del entrenamiento en septiembre de 2012

- **Fecha:** 2026-10-30
- **Contexto:** 7 variables de buró tenían un PSI de ~0,84 entre entrenamiento y 2015. Sin contar los nulos, el PSI era < 0,01: los clientes no cambiaron. Lending Club empezó a reportar esas variables entre marzo y septiembre de 2012, así que de ene a ago-2012 vienen vacías, y en 2015 nunca hay nulos. Ver `notebooks/03_pipeline_estabilidad.ipynb`.
- **Decisión:** el entrenamiento empieza en **sep-2012**. Se excluyen 23.209 préstamos (7,6% del entrenamiento original; 14,1% de malos). Con la población final, el PSI máximo de todas las variables es 0,036. Queda: entrenamiento sep-2012 a 2014 con 283.253 préstamos y 13,2% de malos; OOT 2015 con 283.026 préstamos y 14,9% de malos.
- **Consecuencia:** al recalcular el IV con la nueva población, se aplica la regla de la decisión 002 (IV ≥ 0,02) a la lista de la logística. Salen `revol_util` (0,019) e `inq_last_6mths` (0,0198) y entran `antig_credito_meses` (0,0205) y `num_actv_rev_tl` (0,0213). Todas siguen en los modelos de árboles.
- **Alternativas consideradas:** B) dejar esos préstamos con los nulos (los árboles podrían aprender un patrón "nulo = 2012" que no se repite); C) sacar las 7 variables (se pierden variables con buen IV).
- **Por qué (en mis palabras):** A, para tener una población homogénea sin nulos artificiales.

---

## 004 — Variables con signo contrario a la lógica en la logística

- **Fecha:** 2026-10-30
- **Contexto:** con las 16 variables, tres coeficientes sobre WoE salieron positivos, es decir, contrarios a la lógica de riesgo, por multicolinealidad: `tot_cur_bal` (correlación 0,89 con `avg_cur_bal`, VIF 7,1), `bc_util` (correlación 0,79 con `percent_bc_gt_75`) y `antig_credito_meses` (duplica `mo_sin_old_rev_tl_op`). Ver `notebooks/04_logistica_scorecard.ipynb`.
- **Decisión:** quitar las 3 de la **regresión logística**; queda con 13 variables, todas con signo lógico. Los modelos de árboles las conservan. El AUC fuera de tiempo pasa de 0,6534 a 0,6526 (−0,0008).
- **Alternativas consideradas:** B) dejarlas para ganar una milésima de AUC; C) quitar la otra variable de cada par correlacionado.
- **Por qué (en mis palabras):** A, para que todos los signos sean lógicos y explicables. Sobre la antigüedad crediticia, no creo que sea una variable aceptada ya que va en contra de la lógica.

---

## 005 — Ajuste de hiperparámetros con optimización bayesiana

- **Fecha:** 2026-10-30
- **Contexto:** con los parámetros por defecto, en OOT 2015, XGBoost (AUC 0,6792) y LightGBM (0,6788) empatan con el benchmark `int_rate` (0,6784); la red neuronal obtiene 0,6737 y Random Forest 0,6649. Para que la comparación sea justa, todos los retadores deben recibir el mismo esfuerzo de ajuste.
- **Decisión:**
  - **Optimización bayesiana** con Optuna (muestreador TPE), **25 ensayos por modelo**, mismo presupuesto para los 4 retadores.
  - **Para mitigar el sobreajuste a la validación:** el objetivo es el AUC **promedio de 2 cortes temporales** (validar jul–sep 2014 ajustando hasta jun-2014, y validar oct–dic 2014 ajustando hasta sep-2014). Un hiperparámetro tiene que funcionar en dos periodos distintos.
  - **2015 (OOT) no se usa en el ajuste**; es el examen final. La brecha entre el AUC de validación y el de OOT de cada modelo mide si hubo sobreajuste.
  - Modelo final: se reentrena con todo el entrenamiento; en los boosting se usa el número promedio de árboles de los cortes, +10% por tener más datos.
  - Código: `src/credit_lab/retadores.py` y `src/credit_lab/ajustar.py`.
- **Alternativas consideradas:** búsqueda aleatoria; dejar los parámetros por defecto; ajustar solo XGBoost y LightGBM (comparación sesgada).
- **Por qué (en mis palabras):** para el ajuste de hiperparámetros usa la optimización bayesiana. El problema de ajustar mucho sobre un solo periodo es que puede causar overfitting, haciendo que en general se pierda poder de predicción.

---

## 006 — Modelo campeón: LightGBM

- **Fecha:** 2026-10-30
- **Contexto:** en 2015, con bootstrap pareado de 300 réplicas, LightGBM (+0,0039 de AUC, IC 95% [+0,0016; +0,0063]) y XGBoost (+0,0038, [+0,0015; +0,0063]) le ganan al benchmark `int_rate` con significancia. Entre ellos es un empate (+0,0001, [−0,0003; +0,0005]). LightGBM captura 21,1% de los malos en el peor decil frente a 21,0% de XGBoost. Ver `notebooks/06_campeon.ipynb`.
- **Decisión:** el campeón es **LightGBM**.
- **Alternativas consideradas:** XGBoost (empate estadístico, entrena en GPU); ensamble LightGBM + XGBoost (más complejo de explicar).
- **Por qué (en mis palabras):** A, por su rapidez en CPU, que generará menos latencia en producción, además de capturar más malos en el peor decil.

---

## 007 — Recalibración del campeón

- **Fecha:** 2026-10-30
- **Contexto:** el campeón ordena bien, pero su PD media en 2015 fue 13,1% frente a 14,9% real (−1,75 puntos), de forma pareja en todos los deciles y en ambos semestres. Aprendió con 2012–2014 (13,2% de mora) y 2015 fue un año más riesgoso.
- **Decisión:** recalibrar con un **desplazamiento del intercepto en escala logit** (+0,1432, que equivale a odds ×1,15), aprendido con el **1er semestre de 2015** y evaluado en el **2º semestre**. Resultado en el 2º semestre: PD media de 12,9% a 14,5% (real 14,7%), Brier de 0,1189 a 0,1185, AUC sin cambio (0,6852). Código: `src/credit_lab/campeon.py`.
- **Alternativas consideradas:** no recalibrar y documentar la subestimación; recalibración isotónica o de Platt (cambian la forma de la curva y piden más datos).
- **Por qué (en mis palabras):** recalibra el campeón.

---

## 008 — Política de aprobación por utilidad esperada

- **Fecha:** 2026-10-30
- **Contexto:** con flujos reales del entrenamiento, un préstamo bueno rinde entre +8% (A1) y +31% (F/G) del monto, y uno malo pierde ~36% en cualquier subgrado; la PD tolerable sube con el precio (~18% en A1, ~48% en G2). En una comparación justa (elegir con ene–jun 2015, evaluar en jul–dic 2015), un corte único de PD (≤ 42,4%) y la utilidad esperada suman lo mismo (+USD 0,66 M y +USD 0,65 M sobre aprobar todo). Ver `notebooks/07_explicabilidad_politica.ipynb`.
- **Decisión:** aprobar si `monto × [(1 − PD) × r_bueno + PD × r_malo] > 0`, con la PD recalibrada del campeón y los rendimientos por subgrado del entrenamiento (`data/processed/rendimientos_subgrado.csv`). Es la regla que usará el agente.
- **Resultado (jul–dic 2015):** 98,2% de aprobación, ROI de 6,30% (vs 6,22% del corte de PD) y 14,4% de mora en los aprobados (vs 14,6%).
- **Alternativas consideradas:** B) corte único de PD (umbral extremo encontrado por búsqueda, frágil); C) utilidad esperada con margen de seguridad (aprueba menos).
- **Por qué (en mis palabras):** A, misma utilidad con mejor ROI y menos mora.

---

## 009 — Alcance: motor de decisión de crédito completo

- **Fecha:** 2026-10-30
- **Contexto:** Neider propuso sumar un modelo de cupo y uno de churn, orquestados por un agente de LangGraph en paralelo junto al riesgo y el pricing. Revisión con los datos:
  - **Cupo:** el monto financiado es igual al pedido en el 100% de los préstamos (Lending Club nunca recortó el cupo), así que una regresión del monto modelaría la demanda, no el riesgo. En su lugar: **motor de cupo óptimo** que simula la PD con distintos montos (el campeón usa `monto_ingreso`) y elige el mayor monto con utilidad esperada positiva.
  - **Churn:** sin `member_id` no se puede saber si un cliente vuelve. El equivalente en crédito es el **prepago**: el 51,5% de los buenos prepaga antes del mes 30, y un bueno que prepaga rinde 13,1% del monto contra 20,5% si llega al final.
  - **Severidad:** el 82,6% de los malos tiene alguna recuperación (7,4% del monto en promedio) → modelo de **LGD**.
- **Decisión:** construir todo. Sesión 8: prepago + LGD; sesión 9: cupo óptimo + pricing; sesión 10: agente LangGraph con especialistas en paralelo (riesgo, prepago, cupo) y un consolidador que redacta el memo, con aprobación humana; sesión 11: MCP, evals y publicación.
- **Alternativas consideradas:** B) sin LGD; C) solo cupo y pricing, sin prepago.
- **Por qué (en mis palabras):** A, para tener un motor de decisión completo.

---

## 010 — Prepago y severidad: construir todas las opciones y elegir con evidencia

- **Fecha:** 2026-10-30
- **Contexto:** solo el 35% de los buenos llega al final del plazo; el que prepaga antes del mes 30 rinde 13,1% del monto contra 20,5%. La LGD clásica es ~89% en todos los grados (poca señal esperada); la pérdida económica varía según cuándo cae el préstamo.
- **Decisión:** construir las **3 opciones de prepago** (A: clasificación del prepago antes del mes 30; B: regresión del rendimiento del préstamo bueno; C: supervivencia en tiempo discreto del mes de pago total) y las **2 de severidad** (A: LGD clásica de Basilea; B: pérdida económica por cliente), y elegir la mejor de cada familia.
- **Criterio fijado antes de entrenar:** todas terminan en un `r_bueno` o `r_malo` por cliente y se comparan en 2015 con (1) MAE contra el valor real, (2) correlación de Spearman y (3) utilidad de la política en jul–dic 2015 (elegir con ene–jun). Línea base a vencer: el promedio por subgrado actual.
- **Variables:** las 31 de los árboles **más `int_rate`**. No contradice la decisión 001: aquí no se predice riesgo sino comportamiento dado el precio, y la tasa es el incentivo a refinanciar.
- **Por qué (en mis palabras):** hacer las 3 de prepago y las 2 de severidad; hacemos todos los modelos y después, con una evaluación pertinente, nos quedamos con la mejor opción para este business case, porque me interesa aprender de todos.

---

## 011 — Prepago por supervivencia y severidad por pérdida económica

- **Fecha:** 2026-10-30
- **Contexto (2015, fuera de tiempo):**
  - **Prepago (buenos):** supervivencia con el mejor RMSE (0,0497 vs 0,0540–0,0558), el mejor Spearman (0,667) y el nivel más cercano al real (14,7% vs 14,2%; la línea base asumía 16,1%). La regresión gana en MAE por poco, pero tiene el peor RMSE (objetivo L1). AUC del clasificador de prepago: 0,633.
  - **Severidad (malos):** la LGD clásica no mejora al promedio (R² ≈ 0; el 92% de las LGD está entre 80% y 100%) y en la política **empeora la utilidad de forma significativa** (−USD 0,20 M, IC 95% [−0,36; −0,04]). La pérdida económica tiene algo de señal (Spearman 0,12 vs 0,03).
  - **Política jul–dic 2015:** supervivencia + pérdida económica es la mejor en utilidad (+USD 0,25 M vs decisión 008), ROI (6,37%) y mora (14,3%), aunque la ventaja no es significativa (IC 95% [−0,12; +0,69]).
- **Decisión:** el motor usa **prepago por supervivencia (C)** y **severidad por pérdida económica (B)**. Ver `notebooks/08_prepago_severidad.ipynb`.
- **Alternativas consideradas:** mantener los promedios de la decisión 008; solo supervivencia con severidad promedio.
- **Por qué (en mis palabras):** A, la mejor en todas las métricas y sirve para pricing. Además pedí incluir el RMSE junto al MAE en las evaluaciones.

---

## 012 — Motor de cupo y pricing: simulador + regresiones de referencia

- **Fecha:** 2026-10-30
- **Contexto:** Neider propuso modelar con regresiones el monto a prestar y la tasa. Con los datos: una regresión del monto aprende la **demanda** (lo que pide el cliente; monto financiado = pedido en el 100%) y una de la tasa aprende **el precio de Lending Club** (145 tasas salidas de 35 subgrados; ±0,6 puntos dentro de cada subgrado). Ninguna dice cuánto es seguro prestar ni cuál es el precio correcto según el riesgo propio.
- **Decisión:**
  - **Simulador monto × tasa** por cliente: la PD del campeón cambia con el monto (`monto_ingreso`), la supervivencia y la pérdida económica cambian con la tasa → utilidad esperada de cada combinación.
  - **Capacidad de pago:** cuota mensual ≤ **20%** del ingreso mensual.
  - **Salidas:** monto máximo con utilidad positiva que cumple la capacidad de pago, y tasa mínima rentable para el monto pedido.
  - **Regresiones de referencia:** monto que pediría el cliente (demanda) y tasa que cobraría Lending Club (precio de mercado), para comparar con lo recomendado.
- **Supuestos documentados:** la PD no reacciona a la tasa (el modelo de riesgo excluye `int_rate` por diseño, decisión 001); no hay datos de aceptación, así que el precio se calcula sin la elasticidad del cliente.
- **Alternativas consideradas:** solo las regresiones (replican demanda y precio de Lending Club); solo el optimizador.
- **Por qué (en mis palabras):** A, con límite de cuota sobre ingreso del 20%.

---

## 013 — Umbrales de pricing: equilibrio (0%) y objetivo (4%)

- **Fecha:** 2026-10-30
- **Contexto:** la utilidad del motor es **bruta**: no descuenta fondeo ni costos operativos (no están en los datos). Con ROI objetivo de 0%, la tasa mínima cae en el piso de la grilla (5%) en el 66% de los casos. Backtest en 10.000 préstamos de jul–dic 2015:

  | ROI objetivo | Tasa mínima mediana | En el piso de 5% | LC cobró por debajo | ROI real de esos | ROI real del resto |
  |---|---|---|---|---|---|
  | 0% | 5,0% | 66% | 3,2% | −0,6% | 6,1% |
  | 4% | 7,5% | 18% | 16,3% | 2,8% | 6,5% |

- **Decisión:** dos umbrales. **Tasa de equilibrio** (ROI esperado ≥ 0%) como piso de rechazo: por debajo, el préstamo pierde dinero esperado. **Tasa sugerida** (ROI esperado ≥ **4%**) como precio: cubre lo que la utilidad bruta no descuenta.
- **Alternativas consideradas:** objetivo de 3% (31% de clientes en el piso) o de 5% (marca a 1 de cada 4 préstamos).
- **Por qué (en mis palabras):** B, ROI objetivo del 4%.

---

## 014 — Efecto causal de la tasa: sesión 10 completa

- **Fecha:** 2026-10-30
- **Contexto:** el motor supone que la PD no reacciona a la tasa (decisión 012), pero el backtest lo contradice: con más de 6 puntos de sobreprecio la mora sube a 18,6%. Si la tasa empuja la mora (el cliente paga una cuota más pesada) o atrae a peores clientes (selección adversa), la tasa sugerida subestima el riesgo. Experimento natural: dentro de un mismo subgrado, el **96,3%** de la variación de la tasa la explica el mes de originación, porque Lending Club cambiaba sus tablas de tasas en meses distintos para cada subgrado.
- **Decisión:** sesión 10 completa.
  - Efectos fijos de dos vías (subgrado + mes), que equivale a un diff-in-diff con los cambios de tabla.
  - DML manual con LightGBM y cross-fitting.
  - Heterogeneidad del efecto por segmento.
  - Integración al motor: PD(tasa) → tasa que maximiza el ROI.
  - El agente LangGraph pasa a la sesión 11 y la publicación a la 12.
- **Limitaciones documentadas:** no hay datos de aceptación (no se observa a quien rechazó la oferta), así que el efecto mezcla el impacto directo de la cuota con la selección adversa; para el motor ambos importan igual. EconML y DoubleML no están instalados: el DML se programa a mano.
- **Alternativas consideradas:** solo DML (B); dejarlo para después (C).
- **Por qué (en mis palabras):** A, sesión 10 completa.

---

## 015 — El efecto de la tasa no se identifica: PD fija + escenario de estrés

- **Fecha:** 2026-10-30
- **Contexto (entrenamiento sep-2012–2014, IC 95% agrupado por subgrado):**

  | Especificación | Mora por +1 pt de tasa | Placebo: tasa del mes siguiente |
  |---|---|---|
  | Correlación simple | +1,51 pts | — |
  | Dos vías (subgrado + mes) | −0,22 | — |
  | Dos vías + perfil | −0,45 (con placebo: +0,95) | −1,98 [−2,57; −1,39] |
  | DML con LightGBM | −0,06 [−0,61; +0,48] | −1,46 [−2,07; −0,84] |

  Muestra: 276.342 préstamos con dato del mes siguiente.

  - La correlación tasa–mora es selección: desaparece al comparar dentro del subgrado.
  - El placebo falla con todos los métodos. Un mes antes de cada subida de tasa el subgrado ya traía ~6,7 pts menos de FICO por punto de tasa: Lending Club cambiaba sus tablas reaccionando a cómo se movía cada subgrado. Los efectos ≈ 0 no son causales.
  - La pendiente del motor (+1,0% de ROI por punto de tasa) es más conservadora que las estimaciones (+1,3% a +1,4%).
  - Backtest jul–dic 2015: donde Lending Club cobró más que la tasa de mercado del perfil, la mora real supera a la PD del campeón. Es información de Lending Club, no un efecto causal demostrado.
- **Decisión:** reportar el resultado negativo y mantener la PD fija ante la tasa. El motor agrega:
  - **Tasa de mercado** recalibrada en nivel con ene–jun 2015 (−1,51 pts, igual que la decisión 007).
  - **Escenario de estrés** opcional: la PD sube +1 pt por cada punto que la tasa supera la de mercado (la mayor estimación puntual, +0,95). Backtest jul–dic 2015: el 4,9% de los clientes no llega al ROI de 4% con ninguna tasa bajo estrés; su ROI real fue −1,3% y su mora 31,5% (contra 5,9% y 14,5% del total).
  - **Bandera "fuera de rango"**: la tasa se aleja más de 1 pt de la de mercado; las tablas solo movieron la tasa ±0,74 pts.
  - No se estima heterogeneidad (sin identificación no tiene sentido) ni la "tasa que maximiza el ROI" (requiere datos de aceptación). Ver `notebooks/10_inferencia_causal.ipynb`.
- **Alternativas consideradas:** buscar otro diseño (event study, B); integrar igual la estimación del DML (C).
- **Por qué (en mis palabras):** A, resultado negativo + estrés.

---

## 016 — Agente 100% IA: especialistas con LLM + comité + auditor

- **Fecha:** 2026-10-31
- **Contexto:** la decisión 009 planteaba un agente con aprobación humana al final. Neider pidió un flujo **100% IA**, sin humano. Se compararon tres arquitecturas: números por código con un Claude que decide y otro que audita (A), especialistas con LLM (B) y un solo agente ReAct (C).
- **Decisión:** **B**, en LangGraph:
  - **Tres especialistas Claude en paralelo**, cada uno con sus herramientas sobre los modelos del repo: riesgo (PD + razones SHAP), economía (prepago + pérdida) y cupo/pricing (monto máximo, tasas, estrés, fuera de rango).
  - **Comité de IA** que consolida y decide: aprobar, rechazar o contraofertar con monto y tasa.
  - **Auditor de IA** que reemplaza al humano: verifica que los números del memo coincidan con los de las herramientas y que se cumplan las reglas. Si falla, devuelve el caso al comité.
  - **Reglas duras** que ningún LLM puede saltarse: cuota ≤ 20% del ingreso (decisión 012) y rechazo si el ROI esperado < 0% (decisión 013).
  - Modelo: **Claude Sonnet 5.5** (`claude-sonnet-5-5`).
- **Trade-offs aceptados:** unas 5 llamadas por solicitud (≈3× costo y latencia frente a A) y más puntos donde pueden aparecer números inventados; se mitigan con el auditor, las reglas duras y las evals de la sesión 12.
- **Alternativas consideradas:** A (números por código) y C (un solo agente ReAct).
- **Por qué (en mis palabras):** B, especialistas con LLM; quiero que todo sea full IA, sin una persona al final del flujo.
- **Pendiente resuelto (2026-10-31):** la prueba con Claude real queda pendiente porque no se pudo cargar saldo a la API. El flujo está probado con un cliente simulado (4 tests) y las herramientas con los modelos reales. Ese mismo día se cargó saldo y el agente corrió con Claude real; ver la decisión 018.

---

## 017 — Sesión 12: servidor MCP del motor

- **Fecha:** 2026-10-31
- **Contexto:** la prueba del agente con Claude real quedó pendiente (sin saldo en la API). Se evaluó qué avanzar sin necesitarla: servidor MCP, evals, README final con release y demo con el cliente simulado.
- **Decisión:** la sesión 12 es el **servidor MCP** (`src/credit_lab/servidor_mcp.py`, SDK `mcp` 2.x).
  - **Herramientas:** `buscar_solicitud`, `consultar_riesgo`, `consultar_economia`, `consultar_cupo_pricing` y `simular_oferta`, con salida estructurada. Más un recurso `credito://politica` con las reglas del motor.
  - **Las mismas herramientas que usan los especialistas del agente:** son código puro y no necesitan API key.
  - **Datos:** préstamos de la muestra pública de jul–dic 2015, identificados por su posición en esa muestra, nunca por un id de Lending Club.
- **Alternativas consideradas:** evals, README final + release, demo sin API. Quedan para la sesión 13.
- **Por qué (en mis palabras):** servidor MCP.

---

## 018 — Registro de ejecuciones en SQLite + trazas en LangSmith

- **Fecha:** 2026-10-31
- **Contexto:** Neider propuso guardar en una base las respuestas de los modelos, para evaluar después su desempeño, y sumar LangSmith para medir el desempeño agéntico. Ventaja de este proyecto: en 2015 se conoce el resultado real de cada préstamo, así que se puede medir si las decisiones del agente rinden, no solo si redacta bien.
- **Decisión:**
  - **SQLite** (`src/credit_lab/registro.py`, base en `data/agente/ejecuciones.db`):
    - Tabla `ejecuciones`: decisión, monto, tasa, auditoría, intentos, hallazgos, memo, llamadas, tokens, costo, latencia y **resultado real** (malo y ROI), que el agente nunca ve.
    - Tabla `pasos`: cada respuesta de los especialistas (con sus llamadas a herramientas), del comité y del auditor, por intento.
    - `resumen()` reporta mora y ROI real por decisión, tasa de aprobación de la auditoría, costo y latencia.
  - **LangSmith:** se activa solo si existe `LANGSMITH_API_KEY` (proyecto `credit-risk-agent-lab`). Traza el grafo de LangGraph y cada llamada a Claude (`wrap_anthropic`); sin la key, el agente corre igual.
  - **La base no se sube a git;** más adelante se puede publicar un resumen agregado de las evals.
- **Alternativas consideradas:** solo SQLite; solo LangSmith; publicar la base.
- **Por qué (en mis palabras):** SQLite + LangSmith, fuera de git y en una sesión aparte.
- **Prueba real (2026-10-31):** 3 préstamos de jul–dic 2015 registrados en SQLite y trazados en LangSmith (región US). Cada solicitud usa 8 llamadas, cuesta ~USD 0,07 y tarda ~30 s. En un caso el auditor rechazó el primer memo por inconsistencias de argumentación (no de números) y el comité lo corrigió.

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

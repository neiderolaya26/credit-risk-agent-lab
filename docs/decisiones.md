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

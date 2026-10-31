# credit-risk-agent-lab

> 🚧 En construcción. Proyecto de aprendizaje construido sesión a sesión.

**Motor de decisión de crédito** de punta a punta con datos públicos de **Lending Club**: se entrenan y comparan **6 modelos de riesgo** (regresión logística, Random Forest, XGBoost, LightGBM y dos redes neuronales) contra el score de Lending Club; el campeón se combina con modelos de **prepago** y **severidad (LGD)**, un **motor de cupo** y **pricing**, y todo se orquesta con un **agente de IA en LangGraph** (especialistas en paralelo) que explica cada decisión y redacta el memo de crédito para un analista humano.

## Pregunta de negocio

_🟡 Borrador por validar con Neider:_ ¿Puede un modelo entrenado solo con datos públicos de la solicitud **ordenar el riesgo mejor que el score de Lending Club**, y convertir esa ventaja en **mejores decisiones de aprobación, cupo y precio**?

## Diagrama

```mermaid
flowchart LR
    D[Lending Club<br/>datos públicos] --> F[Features<br/>WoE / IV]
    F --> M1[Logística]
    F --> M2[Random Forest]
    F --> M3[XGBoost]
    F --> M4[LightGBM]
    F --> M5[Red neuronal]
    M1 & M2 & M3 & M4 & M5 --> E[Evaluación<br/>AUC · KS · PSI · calibración · utilidad]
    E --> C[Campeón<br/>LightGBM recalibrado]
    C --> A[Agente LangGraph<br/>especialistas en paralelo]
    PP[Prepago] --> A
    L[Severidad LGD] --> A
    A --> CU[Cupo + pricing]
    CU --> H[Memo para<br/>analista humano]
```

## Hoja de ruta

| Sesión | Tema | Entregable |
|---|---|---|
| 1 ✅ | Entorno + datos + definir el target | Datos descargados, decisión 001 |
| 2 ✅ | EDA y fuga de información (leakage) | Notebook con mis conclusiones, decisión 002 |
| 3 ✅ | Features: WoE, IV, partición fuera de tiempo | Pipeline de features + tests, decisión 003 |
| 4 ✅ | Línea base: regresión logística / scorecard | Modelo 1 + métricas, decisión 004 |
| 5 ✅ | Random Forest, XGBoost, LightGBM, red neuronal (scikit-learn y PyTorch) | Retadores con ajuste bayesiano, decisión 005 |
| 6 ✅ | Comparación: significancia (bootstrap), calibración, utilidad por punto de corte | Campeón LightGBM recalibrado, decisiones 006-007 |
| 7 ✅ | Explicabilidad (SHAP) y política de aprobación por utilidad esperada | Razones por préstamo, decisión 008 |
| 8 ✅ | Modelo de prepago (el "churn" del crédito) y severidad (LGD) | Supervivencia + pérdida económica, decisiones 009-011 |
| 9 | Motor de cupo óptimo y pricing (tasa mínima rentable) | Monto y tasa recomendados por cliente |
| 10 | Agente LangGraph: especialistas en paralelo + memo + aprobación humana | Agente funcionando |
| 11 | Servidor MCP, evals, demo y publicación | Release v1.0.0 |

## Decisiones y trade-offs

Ver [docs/decisiones.md](docs/decisiones.md).

## Cómo correrlo

_(Se completa al final.)_

## Licencia

MIT

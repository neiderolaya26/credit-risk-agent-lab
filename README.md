# credit-risk-agent-lab

> ✅ v1.0.0 · Proyecto construido sesión a sesión, con cada decisión registrada en [docs/decisiones.md](docs/decisiones.md).

**Motor de decisión de crédito** de punta a punta con datos públicos de **Lending Club**: se entrenan y comparan **6 modelos de riesgo** (regresión logística, Random Forest, XGBoost, LightGBM y dos redes neuronales) contra el score de Lending Club; el campeón se combina con modelos de **prepago** y **severidad (LGD)**, un **motor de cupo** y **pricing** validado con **inferencia causal** (efectos fijos, placebo y DML sobre el efecto de la tasa en la mora), y todo se orquesta con un **agente de IA en LangGraph** (especialistas en paralelo) que explica cada decisión, la audita y redacta el memo de crédito sin intervención humana.

## Pregunta de negocio

¿Se puede, **solo con un dataset público**, construir de punta a punta un sistema que tome **mejores decisiones de aprobación, cupo y precio**, y que **automatice la decisión end to end** con modelamiento avanzado, LLM y agentes?

## Resultados clave

| Pieza | Resultado (fuera de tiempo, 2015) | Decisión |
|---|---|---|
| Riesgo | El campeón LightGBM (AUC 0,6823) le gana al score de Lending Club (0,6784) solo con variables públicas de la solicitud; IC 95% de la diferencia [+0,0016; +0,0063] | 006–007 |
| Aprobación | La política por utilidad esperada logra la misma utilidad que un corte por PD, con mejor ROI (6,30%) y menos mora | 008 |
| Prepago y severidad | La supervivencia en tiempo discreto predice el rendimiento mejor que las alternativas (RMSE 0,0497); la LGD clásica no tiene señal | 011 |
| Cupo y precio | Los préstamos que Lending Club cobró por debajo de la tasa sugerida rindieron 2,8% frente a 6,7%; los que pidieron más que el monto máximo, 1,8% con 30% de mora | 012–013 |
| Inferencia causal | La relación tasa–mora es selección, no causa: desaparece dentro del subgrado, pero el placebo falla, así que el efecto no queda identificado. El escenario de estrés detecta un 4,9% de clientes sin tasa viable que en la realidad rindieron −1,3% con 31,5% de mora | 014–015 |
| Agente 100% IA | Con Haiku 5.5 en 30 casos con resultado real: 100% de reglas duras, 94% de números respaldados por herramientas, 93% de auditorías aprobadas al primer intento y 97% en el juez (Opus 5.5), a USD 0,007 por solicitud | 016–020 |

## Diagrama

```mermaid
flowchart LR
    D[Lending Club<br/>datos públicos] --> M[6 modelos de riesgo<br/>campeón LightGBM]
    D --> PP[Prepago<br/>supervivencia]
    D --> L[Pérdida<br/>económica]
    IC[Inferencia causal<br/>estrés de tasa] --> CU
    M & PP & L --> CU[Motor de cupo<br/>y pricing]
    CU --> ESP[Agente LangGraph<br/>3 especialistas Claude]
    ESP --> CO[Comité IA]
    CO --> RD[Reglas duras]
    RD --> AU[Auditor IA]
    AU --> DEC[Decisión + memo]
    CU -. herramientas .-> MCP[Servidor MCP]
    DEC --> REG[SQLite + LangSmith<br/>+ evals]
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
| 9 ✅ | Motor de cupo óptimo y pricing (tasa mínima rentable) | Monto máximo y tasas de equilibrio y sugerida, decisiones 012-013 |
| 10 ✅ | **Inferencia causal**: efecto de la tasa en la mora (efectos fijos, placebo, DML) | Resultado negativo + escenario de estrés, decisiones 014-015 |
| 11 ✅ | Agente LangGraph 100% IA: especialistas Claude en paralelo + comité + auditor | Agente funcionando con Claude Sonnet 5.5 (~USD 0,07 y ~30 s por solicitud), decisión 016 |
| 12 ✅ | Servidor MCP con las herramientas del motor (riesgo, economía, cupo/pricing, simulación) | Servidor `credit-lab`, decisión 017 |
| 13 ✅ | Registro de ejecuciones en SQLite + trazas en LangSmith (desempeño agéntico) | Ejecuciones registradas con su resultado real y trazadas, decisión 018 |
| 14 ✅ | **Evals del agente**: 30 casos con resultado real, calificación programada + juez Opus, versiones v0 → v2 con Haiku 5.5 | Concordancia 83%, fidelidad 94%, juez 97%, USD 0,007 por solicitud, decisión 019 |
| 15 ✅ | Regla 2b con ROI ≥ 4%, contraofertas verificables (v3), demo y publicación | Notebook 11, página del proyecto y release v1.0.0, decisión 020 |
| Próximo | Comparar Haiku y Sonnet 5.5 en los mismos 30 casos con la v3 | Tabla comparativa de calidad y costo |

## Decisiones y trade-offs

Ver [docs/decisiones.md](docs/decisiones.md).

## Cómo correrlo

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m credit_lab.ingest                          # descarga Lending Club (requiere token de Kaggle)
pytest                                               # tests
```

### Servidor MCP

Expone las herramientas del motor a cualquier cliente MCP. No necesita API key. Por ejemplo, en Claude Code:

```bash
claude mcp add credit-lab --env PYTHONPATH=<ruta-del-repo>/src -- <ruta-del-repo>/.venv/Scripts/python -m credit_lab.servidor_mcp
```

| Herramienta | Qué devuelve |
|---|---|
| `buscar_solicitud` | Índice y resumen de un préstamo de jul–dic 2015 |
| `consultar_riesgo` | PD del campeón y razones SHAP |
| `consultar_economia` | Rendimiento si sale bueno o malo, prepago y ROI esperado |
| `consultar_cupo_pricing` | Monto máximo, tasas de equilibrio y sugerida, estrés y capacidad de pago |
| `simular_oferta` | Métricas para otro monto y tasa |

### Agente

```bash
python -m credit_lab.agente --semilla 0    # requiere ANTHROPIC_API_KEY en .env (no se sube a git)
```

Cada ejecución queda registrada en `data/agente/ejecuciones.db` (SQLite, fuera de git), con el resultado real del préstamo para evaluar las decisiones:

```python
from credit_lab import registro
with registro.conectar() as con:
    print(registro.resumen(con))   # mora y ROI real por decisión, auditoría, costo y latencia
```

### Evals del agente

```bash
python -m credit_lab.evals casos                              # elige los 30 casos (gratis)
python -m credit_lab.evals correr --modelo claude-haiku-5-5   # corre el agente y el juez sobre los casos pendientes
python -m credit_lab.evals reporte                            # evals/agente/<modelo>/<versión>/reporte.md
```

Resultados y trazas publicados en [evals/agente](evals/agente).

Para ver las trazas del grafo en LangSmith, agrega `LANGSMITH_API_KEY` al `.env`; el proyecto se llama `credit-risk-agent-lab`.

## Licencia

MIT

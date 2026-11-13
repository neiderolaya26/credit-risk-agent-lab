---
license: mit
library_name: lightgbm
pipeline_tag: tabular-classification
tags:
  - credit-risk
  - probability-of-default
  - lightgbm
  - survival-analysis
  - lending-club
  - tabular
metrics:
  - roc_auc
---

# Credit risk · Lending Club (campeón LightGBM y modelos del motor)

Modelos del repositorio [credit-risk-agent-lab](https://github.com/neiderolaya26/credit-risk-agent-lab), un sistema de decisión de crédito de punta a punta construido solo con datos públicos de Lending Club. Demo en vivo: [motor de decisión en Streamlit](https://credit-risk-agent-lab.streamlit.app/).

## Archivos

| Archivo | Qué es |
|---|---|
| `campeon.joblib` | PD a 36 meses: preprocesador + LightGBM y el desplazamiento logit de la recalibración (decisiones 006–007) |
| `prepago_c_supervivencia.joblib` | Rendimiento si el préstamo sale bueno: prepago con supervivencia en tiempo discreto (decisión 011) |
| `severidad_b_economica.joblib` | Rendimiento si sale malo: pérdida económica (decisión 011) |
| `referencia_tasa.joblib` | Tasa de mercado del perfil, recalibrada con ene–jun 2015 (decisión 015) |

Son objetos de Python guardados con joblib que usan clases del paquete `credit_lab` del repositorio: instálalo para cargarlos. Como todo pickle, cárgalos solo desde una fuente en la que confíes.

```python
from huggingface_hub import snapshot_download
snapshot_download("neiderolaya26/credit-risk-lending-club", local_dir="modelos")
# con el repositorio clonado y PYTHONPATH=src:
from credit_lab.motor import Motor
motor = Motor()          # lee modelos/ en la raíz del repo
```

## Datos y validación

- **Población:** préstamos a 36 meses de Lending Club; entrenamiento sep-2012 a 2014 y validación fuera de tiempo en 2015. Solo variables de la solicitud: sin la tasa ni el score de Lending Club, sin identificadores, texto libre ni código postal.
- **Variable objetivo:** `malo` = Charged Off o Default.
- **Ajuste:** Optuna (TPE) con 2 cortes temporales dentro del entrenamiento.

## Desempeño (2015, fuera de tiempo)

| Métrica | Valor |
|---|---|
| AUC del campeón | 0,6823 |
| AUC del score de Lending Club (su tasa) | 0,6784 |
| IC 95% de la diferencia (bootstrap) | [+0,0016; +0,0063] |
| Calibración por grupo de vivienda y región (jul–dic 2015) | PD media a ±0,7 pts de la mora real |
| Supervivencia (prepago) | RMSE 0,0497 |

## Uso previsto y límites

- Uso previsto: investigación y demostración de un motor de decisión (PD, economía, política y razones). **No** está pensado para decidir créditos reales.
- Solo préstamos que Lending Club aprobó (2012–2015): no hay solicitudes rechazadas ni datos de aceptación de ofertas.
- El efecto causal de la tasa en la mora no se pudo identificar: la PD no reacciona a la tasa y el escenario de estrés es un supuesto.
- Equidad: los datos no traen atributos protegidos. La antigüedad crediticia es un proxy de edad: no se usa como razón; quitarla cuesta −0,0032 de AUC.

La ficha completa del modelo y del agente está en [docs/fichas.md](https://github.com/neiderolaya26/credit-risk-agent-lab/blob/main/docs/fichas.md).

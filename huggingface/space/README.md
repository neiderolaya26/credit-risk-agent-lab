---
title: Motor de decisión de crédito
emoji: 🏦
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: PD, política de crédito y razones TreeSHAP sobre Lending Club
---

# Motor de decisión de crédito · Lending Club

Demo del repositorio [credit-risk-agent-lab](https://github.com/neiderolaya26/credit-risk-agent-lab): elige una solicitud real de jul–dic 2015, que el modelo nunca vio, o cambia sus datos, y mira qué decide la política.

- **PD:** LightGBM recalibrado, AUC 0,6823 fuera de tiempo frente a 0,6784 del score de Lending Club.
- **Economía:** prepago por supervivencia y pérdida económica → ROI esperado.
- **Política (código, sin LLM):** aprobar con ROI ≥ 4% y cuota ≤ 20% del ingreso; si no, contraoferta bajando primero el monto y después subiendo la tasa (máximo +2 pts); si no, rechazar.
- **Razones:** TreeSHAP para el analista y un catálogo fijo de máximo 4 razones para el cliente, sin proxies de edad.

En el repositorio, un agente LangGraph con Claude analiza el caso, opina y redacta el memo, y un analista firma lo que no es una aprobación clara. Este Space no llama a ningún LLM.

Datos públicos de Lending Club. Proyecto personal: no contiene datos de ningún empleador ni cliente.

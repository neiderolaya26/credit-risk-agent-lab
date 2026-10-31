# data/processed

Datos derivados que sí se publican: son agregados, sin identificadores de préstamos ni de personas.

| Archivo | Qué es | Origen |
|---|---|---|
| `rendimientos_subgrado.csv` | Por cada subgrado de Lending Club (A1–G5): rendimiento promedio sobre el monto de los préstamos buenos (`r_bueno`) y malos (`r_malo`), PD de equilibrio y número de préstamos. Lo usa la política de aprobación (decisión 008) | Entrenamiento sep-2012 a 2014, 36 meses · `python -c "from credit_lab.politica import guardar_rendimientos; guardar_rendimientos()"` |

**Fuente:** Lending Club, préstamos aprobados 2007–2018 ([Kaggle `wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club), CC0).

**Ojo:** los subgrados con pocos préstamos (ver columna `prestamos`; G5 tiene 6) tienen rendimientos poco confiables. El agente debe advertirlo cuando evalúe un préstamo de esos subgrados.

"""Entrena las opciones de prepago y severidad (decisión 010) y guarda sus
predicciones para TODOS los préstamos de 2015 (la política las necesita para
buenos y malos, porque al decidir no se sabe cuál será cuál).

Uso:
    python -m credit_lab.entrenar_economia
"""

import time

import joblib
import pandas as pd

from credit_lab import prepago, severidad
from credit_lab.poblacion import RAIZ, cargar_poblacion

SALIDA = RAIZ / "modelos"

OPCIONES = {
    "prepago_base": (prepago.LineaBase, 0),
    "prepago_a_clasificacion": (prepago.Clasificacion, 0),
    "prepago_b_regresion": (prepago.Regresion, 0),
    "prepago_c_supervivencia": (prepago.Supervivencia, 0),
    "severidad_base": (severidad.LineaBase, 1),
    "severidad_a_lgd": (severidad.LGDClasica, 1),
    "severidad_b_economica": (severidad.PerdidaEconomica, 1),
}


def main() -> None:
    df = cargar_poblacion()
    tr, oot = df[df["muestra"].eq("train")], df[df["muestra"].eq("oot")]
    pred = pd.DataFrame({"malo": oot["malo"]}, index=oot.index)
    for nombre, (clase, malo) in OPCIONES.items():
        t = time.time()
        modelo = clase().fit(tr[tr["malo"].eq(malo)])
        pred[nombre] = modelo.predecir(oot)
        joblib.dump(modelo, SALIDA / f"{nombre}.joblib")
        print(f"{nombre}: {time.time() - t:.0f} s", flush=True)
    if "prepago_a_clasificacion" in OPCIONES:
        pred["prob_prepago"] = joblib.load(SALIDA / "prepago_a_clasificacion.joblib").prob_prepago(oot)
    pred.to_parquet(SALIDA / "predicciones_economia_oot.parquet")
    print("Listo", flush=True)


if __name__ == "__main__":
    main()

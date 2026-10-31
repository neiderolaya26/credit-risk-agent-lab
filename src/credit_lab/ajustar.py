"""Ajusta los retadores, entrena los modelos finales y guarda sus predicciones.

Uso:
    python -m credit_lab.ajustar [n_trials] [modelo ...]

Sin modelos, ajusta los 4. Los resultados de modelos ya ajustados se conservan
en modelos/ajuste.json, así se puede reanudar solo lo que falta.
"""

import json
import sys
import time

import joblib
import pandas as pd

from credit_lab.features import derivar
from credit_lab.metricas import resumen
from credit_lab.poblacion import RAIZ, cargar_poblacion
from credit_lab.retadores import DISPOSITIVO_TORCH, DISPOSITIVO_XGB, ajustar, modelo_final, predecir

SALIDA = RAIZ / "modelos"
MODELOS = ["lgbm", "xgb", "mlp", "torch", "rf"]


def _registro(nombre, n_trials, inicio):
    def callback(estudio, trial):
        print(f"  {nombre} ensayo {trial.number + 1}/{n_trials} · AUC {trial.value:.4f} · "
              f"mejor {estudio.best_value:.4f} · {(time.time() - inicio) / 60:.1f} min", flush=True)
    return callback


def main(n_trials: int = 25, modelos: list[str] = MODELOS) -> None:
    SALIDA.mkdir(exist_ok=True)
    df = derivar(cargar_poblacion())
    tr, oot = df[df["muestra"].eq("train")], df[df["muestra"].eq("oot")]
    pred = pd.read_parquet(SALIDA / "predicciones_oot.parquet")
    ruta_json = SALIDA / "ajuste.json"
    resumen_ajuste = json.loads(ruta_json.read_text()) if ruta_json.exists() else {}
    print(f"XGBoost usa: {DISPOSITIVO_XGB} · PyTorch usa: {DISPOSITIVO_TORCH}", flush=True)

    for nombre in modelos:
        t = time.time()
        estudio = ajustar(nombre, tr, n_trials, callbacks=[_registro(nombre, n_trials, t)])
        estudio.trials_dataframe().to_csv(SALIDA / f"optuna_{nombre}.csv", index=False)
        pre, modelo = modelo_final(nombre, estudio, tr)
        joblib.dump((pre, modelo), SALIDA / f"{nombre}.joblib")
        pred[f"pd_{nombre}"] = predecir(nombre, pre, modelo, oot)

        mejor = estudio.best_trial
        resumen_ajuste[nombre] = {
            "auc_validacion": mejor.value,
            "auc_cortes": mejor.user_attrs["auc_cortes"],
            "n_arboles": mejor.user_attrs.get("n_arboles"),
            "params": mejor.params,
            "auc_oot": resumen(oot["malo"], pred[f"pd_{nombre}"])["auc"],
            "minutos": round((time.time() - t) / 60, 1),
            "dispositivo": {"xgb": DISPOSITIVO_XGB, "torch": DISPOSITIVO_TORCH}.get(nombre, "cpu"),
        }
        print(nombre, json.dumps(resumen_ajuste[nombre]), flush=True)
        pred.to_parquet(SALIDA / "predicciones_oot.parquet")
        ruta_json.write_text(json.dumps(resumen_ajuste, indent=2))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    main(n, sys.argv[2:] or MODELOS)

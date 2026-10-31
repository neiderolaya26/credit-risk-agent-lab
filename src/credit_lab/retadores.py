"""Modelos retadores y su ajuste con optimización bayesiana (decisión 005).

- Optuna con TPE (Tree-structured Parzen Estimator), mismo presupuesto por modelo.
- Validación temporal con 2 cortes dentro del entrenamiento para no sobreajustar
  a un solo periodo:
    corte 1: ajusta hasta jun-2014, valida jul–sep 2014
    corte 2: ajusta hasta sep-2014, valida oct–dic 2014
  El objetivo es el AUC promedio de los dos cortes.
- 2015 (OOT) no se usa en ningún momento del ajuste.
"""

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from credit_lab.features import ARBOLES
from credit_lab.red_torch import DISPOSITIVO as DISPOSITIVO_TORCH
from credit_lab.red_torch import RedTorch
from credit_lab.transformadores import preprocesador

CORTES = [("2014-07-01", "2014-10-01"), ("2014-10-01", "2015-01-01")]
SEMILLA = 0
MAX_ARBOLES = 3000
PACIENCIA = 100
RELLENO_RF = -999  # RandomForest recibe los nulos como un valor fuera de rango


def _dispositivo_xgb() -> str:
    """GPU si XGBoost tiene CUDA y hay una tarjeta NVIDIA; si no, CPU.

    En una RTX 5070 Laptop la GPU fue 4,6–7,8× más rápida con el mismo AUC.
    """
    import xgboost
    if not xgboost.build_info().get("USE_CUDA"):
        return "cpu"
    try:
        import numpy as _np
        XGBClassifier(n_estimators=1, device="cuda").fit(_np.zeros((2, 1)), [0, 1])
        return "cuda"
    except Exception:
        return "cpu"


DISPOSITIVO_XGB = _dispositivo_xgb()

FAMILIA = {"rf": "arboles", "xgb": "arboles", "lgbm": "arboles", "mlp": "red", "torch": "red"}


def espacio(nombre: str, trial: optuna.Trial) -> dict:
    if nombre == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 1000, log=True),
            "max_features": trial.suggest_float("max_features", 0.1, 0.6),
        }
    if nombre == "xgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),
        }
    if nombre == "lgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
        }
    if nombre == "mlp":
        capas = trial.suggest_categorical("capas", ["32", "64", "64-32", "128-64", "128-64-32"])
        return {
            "hidden_layer_sizes": tuple(int(c) for c in capas.split("-")),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        }
    if nombre == "torch":
        capas = trial.suggest_categorical("capas", ["64", "128-64", "256-128", "128-64-32", "256-128-64"])
        return {
            "capas": tuple(int(c) for c in capas.split("-")),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
        }
    raise ValueError(nombre)


def construir(nombre: str, params: dict, n_arboles: int | None = None):
    if nombre == "rf":
        return RandomForestClassifier(**params, n_jobs=-1, random_state=SEMILLA)
    if nombre == "xgb":
        extra = ({"n_estimators": n_arboles} if n_arboles else
                 {"n_estimators": MAX_ARBOLES, "early_stopping_rounds": PACIENCIA})
        return XGBClassifier(**params, **extra, eval_metric="auc", tree_method="hist",
                             device=DISPOSITIVO_XGB, n_jobs=-1, random_state=SEMILLA)
    if nombre == "lgbm":
        return lgb.LGBMClassifier(**params, n_estimators=n_arboles or MAX_ARBOLES,
                                  subsample_freq=1, verbose=-1, random_state=SEMILLA)
    if nombre == "mlp":
        return MLPClassifier(**params, early_stopping=True, validation_fraction=0.1,
                             max_iter=200, n_iter_no_change=10, random_state=SEMILLA)
    if nombre == "torch":
        return RedTorch(**params, semilla=SEMILLA)
    raise ValueError(nombre)


def _preparar(nombre, fit, otros):
    pre = preprocesador(FAMILIA[nombre])
    Xf = pre.fit_transform(fit[ARBOLES], fit["malo"])
    Xs = [pre.transform(o[ARBOLES]) for o in otros]
    if nombre == "rf":
        Xf, Xs = Xf.fillna(RELLENO_RF), [x.fillna(RELLENO_RF) for x in Xs]
    return pre, Xf, Xs


def _entrenar(nombre, params, Xf, yf, Xv=None, yv=None, n_arboles=None):
    m = construir(nombre, params, n_arboles)
    if nombre == "xgb" and n_arboles is None:
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        mejor = m.best_iteration + 1
    elif nombre == "lgbm" and n_arboles is None:
        m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric="auc",
              callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
        mejor = m.best_iteration_
    else:
        m.fit(Xf, yf)
        mejor = None
    return m, mejor


def ajustar(nombre: str, train: pd.DataFrame, n_trials: int = 25, callbacks=None) -> optuna.Study:
    """Optimización bayesiana sobre el AUC promedio de los cortes temporales."""
    cortes = []
    for ini, fin in CORTES:
        fit = train[train["issue_d"] < ini]
        val = train[(train["issue_d"] >= ini) & (train["issue_d"] < fin)]
        _, Xf, (Xv,) = _preparar(nombre, fit, [val])
        cortes.append((Xf, fit["malo"], Xv, val["malo"]))

    def objetivo(trial):
        params = espacio(nombre, trial)
        aucs, arboles = [], []
        for Xf, yf, Xv, yv in cortes:
            m, mejor = _entrenar(nombre, params, Xf, yf, Xv, yv)
            aucs.append(roc_auc_score(yv, m.predict_proba(Xv)[:, 1]))
            arboles.append(mejor)
        trial.set_user_attr("auc_cortes", aucs)
        if arboles[0] is not None:
            trial.set_user_attr("n_arboles", int(np.mean(arboles)))
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="maximize", study_name=nombre,
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    estudio.optimize(objetivo, n_trials=n_trials, callbacks=callbacks)
    return estudio


def modelo_final(nombre: str, estudio: optuna.Study, train: pd.DataFrame):
    """Reentrena con todo el entrenamiento y los mejores hiperparámetros.

    En los boosting se usa el número promedio de árboles de los cortes,
    aumentado 10% porque ahora hay más datos.
    """
    mejor = estudio.best_trial
    params = espacio(nombre, optuna.trial.FixedTrial(mejor.params))
    n = mejor.user_attrs.get("n_arboles")
    n = int(n * 1.1) if n else None
    pre, Xt, _ = _preparar(nombre, train, [])
    m, _ = _entrenar(nombre, params, Xt, train["malo"], n_arboles=n)
    return pre, m


def predecir(nombre: str, pre, modelo, df: pd.DataFrame) -> np.ndarray:
    X = pre.transform(df[ARBOLES])
    if nombre == "rf":
        X = X.fillna(RELLENO_RF)
    return modelo.predict_proba(X)[:, 1]

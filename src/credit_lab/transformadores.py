"""Pipeline de preparación de variables, compatible con scikit-learn.

Todo lo que "aprende" de los datos (percentiles, medianas, WoE) se ajusta
solo con entrenamiento y se aplica igual a la muestra fuera de tiempo.

Por familia de modelo:
- logistica: limpieza -> winsorización -> WoE
- arboles:   limpieza -> winsorización -> categóricas a códigos (los nulos se quedan:
             XGBoost, LightGBM y RandomForest los manejan)
- red:       limpieza -> winsorización -> imputación + indicador de nulo -> escala
             y one-hot para categóricas
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_lab.features import ARBOLES, CATEGORICAS, LOGISTICA

CENTINELA = 9_999_999


class Limpieza(BaseEstimator, TransformerMixin):
    """Reglas fijas, no aprende nada de los datos."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                X.loc[X[c] == CENTINELA, c] = np.nan
        if "home_ownership" in X:
            X["home_ownership"] = X["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})
        if "revol_util" in X:
            X["revol_util"] = X["revol_util"].clip(upper=150)
        return X


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta las numéricas a los percentiles 1 y 99 de entrenamiento."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        num = X.select_dtypes("number")
        self.limites_ = num.quantile([self.inferior, self.superior])
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.limites_.columns:
            lo, hi = self.limites_[c]
            X[c] = X[c].clip(lo, hi)
        return X


class WoE(BaseEstimator, TransformerMixin):
    """Reemplaza cada valor por el WoE de su tramo.

    Numéricas: tramos por cuantiles de entrenamiento. Nulos: tramo propio.
    Categoría o tramo no visto en entrenamiento: WoE 0 (riesgo promedio).
    """

    def __init__(self, n_tramos=10, suavizado=0.5):
        self.n_tramos = n_tramos
        self.suavizado = suavizado

    def _tramo(self, x, c):
        if c in self.cortes_:
            t = pd.cut(x, self.cortes_[c], include_lowest=True).astype("string")
        else:
            t = x.astype("string")
        return t.fillna("NULO")

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.cortes_, self.woe_ = {}, {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                _, cortes = pd.qcut(X[c], self.n_tramos, retbins=True, duplicates="drop")
                cortes[0], cortes[-1] = -np.inf, np.inf
                self.cortes_[c] = cortes
            t = self._tramo(X[c], c)
            g = pd.crosstab(t, y).reindex(columns=[0, 1], fill_value=0)
            pb = (g[0] + self.suavizado) / (g[0].sum() + self.suavizado)
            pm = (g[1] + self.suavizado) / (g[1].sum() + self.suavizado)
            self.woe_[c] = np.log(pb / pm).to_dict()
        return self

    def transform(self, X):
        salida = {c: self._tramo(X[c], c).map(self.woe_[c]).astype(float).fillna(0.0)
                  for c in X.columns}
        return pd.DataFrame(salida, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.woe_), dtype=object)


def _numericas(cols):
    return [c for c in cols if c not in CATEGORICAS]


def _categoricas(cols):
    return [c for c in cols if c in CATEGORICAS]


def preprocesador(familia: str) -> Pipeline:
    if familia == "logistica":
        return Pipeline([("limpieza", Limpieza()), ("winsor", Winsorizador()), ("woe", WoE())])

    cols = ARBOLES
    if familia == "arboles":
        cod = ColumnTransformer(
            [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                    encoded_missing_value=-1), _categoricas(cols))],
            remainder="passthrough", verbose_feature_names_out=False,
        )
    elif familia == "red":
        cod = ColumnTransformer(
            [
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("esc", StandardScaler()),
                ]), _numericas(cols)),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100,
                                      sparse_output=False), _categoricas(cols)),
            ],
            verbose_feature_names_out=False,
        )
    else:
        raise ValueError(f"Familia desconocida: {familia}")
    cod.set_output(transform="pandas")
    return Pipeline([("limpieza", Limpieza()), ("winsor", Winsorizador()), ("codificacion", cod)])


def columnas(familia: str) -> list[str]:
    return LOGISTICA if familia == "logistica" else ARBOLES

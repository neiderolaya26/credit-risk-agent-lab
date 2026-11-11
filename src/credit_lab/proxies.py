"""¿Cuánto cuesta en AUC tratar la antigüedad crediticia como proxy de edad? (decisión 023, issue #39)

La antigüedad del historial crediticio crece con la edad, que es un atributo protegido. En la traza c13 una
antigüedad de 545 meses "sube el riesgo", un signo contraintuitivo. Se comparan, con los mismos
hiperparámetros y número de árboles del campeón, entrenando con sep-2012 a 2014 y midiendo en 2015:
    base        → el campeón reentrenado tal cual (referencia comparable);
    monótona    → más antigüedad no puede subir el riesgo (restricción monótona en LightGBM);
    sin proxies → sin antig_credito_meses ni mo_sin_old_rev_tl_op.
El campeón no se cambia: hacerlo obligaría a rehacer el valor, el motor y las evals. Las variables ya se
excluyen de las razones al cliente y del memo (credit_lab.razones).

Uso:
    python -m credit_lab.proxies      # reentrena 3 LightGBM (unos minutos en CPU)
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_lab.campeon import cargar as cargar_campeon
from credit_lab.features import ARBOLES, derivar
from credit_lab.poblacion import cargar_poblacion
from credit_lab.razones import PROXIES_EDAD
from credit_lab.transformadores import preprocesador


def _entrenar(train, test, columnas, restricciones=None, params=None):
    pre = preprocesador("arboles")
    Xf = pre.fit_transform(train[columnas], train["malo"])
    Xt = pre.transform(test[columnas])
    p = dict(params)
    if restricciones:
        p["monotone_constraints"] = [restricciones.get(c, 0) for c in Xf.columns]
        p["monotone_constraints_method"] = "advanced"
    m = lgb.LGBMClassifier(**p).fit(Xf, train["malo"])
    return roc_auc_score(test["malo"], m.predict_proba(Xt)[:, 1])


def comparar() -> pd.DataFrame:
    df = derivar(cargar_poblacion())
    train, test = df[df["muestra"].eq("train")], df[df["muestra"].eq("oot")]
    params = cargar_campeon().modelo.get_params()
    sin = [c for c in ARBOLES if c not in PROXIES_EDAD]
    filas = {
        "base (campeón reentrenado)": _entrenar(train, test, ARBOLES, params=params),
        "monótona (más antigüedad no sube el riesgo)": _entrenar(train, test, ARBOLES,
                                                               {c: -1 for c in PROXIES_EDAD}, params),
        "sin proxies de edad": _entrenar(train, test, sin, params=params),
    }
    t = pd.Series(filas, name="AUC 2015").to_frame()
    t["costo frente a la base"] = t["AUC 2015"] - t.iloc[0, 0]
    return t


def main() -> None:
    pd.set_option("display.float_format", "{:.4f}".format)
    print(comparar())


if __name__ == "__main__":
    main()

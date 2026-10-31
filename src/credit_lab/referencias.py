"""Regresiones de referencia para el motor (decisión 012).

- Monto pedido: cuánto pediría el cliente (demanda). No usa variables que
  contengan el monto (loan_amnt, monto_ingreso).
- Tasa de mercado: qué tasa le cobraría Lending Club (su pricing).

No deciden nada: sirven para comparar lo que recomienda el motor contra lo que
el cliente quiere y lo que cobra el mercado.
"""

import numpy as np

from credit_lab.prepago import Variables, _lgbm, _validacion

SIN_MONTO = ["loan_amnt", "monto_ingreso"]


class MontoPedido:
    def fit(self, df):
        self.var = Variables().fit(df)
        fit, val = _validacion(df)
        self.modelo = _lgbm("regression", self._X(fit), fit["loan_amnt"], self._X(val), val["loan_amnt"])
        return self

    def _X(self, df):
        return self.var.transform(df).drop(columns=SIN_MONTO + ["int_rate"])

    def predecir(self, df):
        return np.clip(self.modelo.predict(self._X(df)), 1000, 35000)


class TasaMercado:
    def fit(self, df):
        self.var = Variables().fit(df)
        fit, val = _validacion(df)
        self.modelo = _lgbm("regression", self._X(fit), fit["int_rate"], self._X(val), val["int_rate"])
        return self

    def _X(self, df):
        return self.var.transform(df).drop(columns=["int_rate"])

    def predecir(self, df):
        return self.modelo.predict(self._X(df))

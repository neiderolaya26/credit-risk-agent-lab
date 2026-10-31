"""Regresiones de referencia para el motor (decisión 012).

- Monto pedido: cuánto pediría el cliente (demanda). No usa variables que
  contengan el monto (loan_amnt, monto_ingreso).
- Tasa de mercado: qué tasa le cobraría Lending Club (su pricing).

No deciden nada: sirven para comparar lo que recomienda el motor contra lo que
el cliente quiere y lo que cobra el mercado.

Decisión 015: la tasa de mercado se recalibra en nivel con ene–jun 2015, como
la PD del campeón (decisión 007), porque Lending Club bajó sus tasas después de
2014 y el motor la usa como referencia del sobreprecio.

Uso:
    python -m credit_lab.referencias   # entrena ambas y guarda modelos/referencia_*.joblib
"""

import joblib
import numpy as np

from credit_lab.poblacion import RAIZ, cargar_poblacion
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
    desplazamiento = 0.0  # puntos de tasa que se suman tras recalibrar

    def fit(self, df):
        self.var = Variables().fit(df)
        fit, val = _validacion(df)
        self.modelo = _lgbm("regression", self._X(fit), fit["int_rate"], self._X(val), val["int_rate"])
        return self

    def _X(self, df):
        return self.var.transform(df).drop(columns=["int_rate"])

    def predecir(self, df):
        return self.modelo.predict(self._X(df))

    def recalibrar(self, df):
        """Nivel: diferencia media entre la tasa cobrada y la predicha en el periodo reciente."""
        self.desplazamiento = float((df["int_rate"] - self.predecir(df)).mean())
        return self

    def predecir_recalibrada(self, df):
        return self.predecir(df) + self.desplazamiento


def main() -> None:
    # Clases importadas del módulo y no de __main__, para que el pickle cargue desde cualquier lado.
    from credit_lab.referencias import MontoPedido, TasaMercado

    df = cargar_poblacion()
    tr = df[df["muestra"].eq("train")]
    s1 = df[df["muestra"].eq("oot") & (df["issue_d"].dt.month <= 6)]
    salida = RAIZ / "modelos"
    joblib.dump(MontoPedido().fit(tr), salida / "referencia_monto.joblib")
    tasa = TasaMercado().fit(tr).recalibrar(s1)
    joblib.dump(tasa, salida / "referencia_tasa.joblib")
    print(f"Referencias guardadas en {salida} · desplazamiento de la tasa = {tasa.desplazamiento:+.3f} pts")


if __name__ == "__main__":
    main()

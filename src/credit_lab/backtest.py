"""Backtest de la política de crédito en jul–dic 2015 (decisión 023).

Sobre una muestra de solicitudes que el modelo nunca vio calcula, una sola vez:
- la política vigente (decisión 023: monto primero, tope de +2 pts y banda de mercado);
- la política anterior (decisión 020: tasa primero, sin tope ni banda), para medir el cambio (issue #36);
- la política y la tasa de equilibrio bajo estrés con varios efectos supuestos (issue #42).

El resultado se guarda en data/interim/backtest_politica.parquet (fuera de git) y lo usan
credit_lab.valor (P&L de la política real, issue #35), la equidad en precio (issue #39) y el
dimensionamiento de la cola del analista (issue #38).

Uso:
    python -m credit_lab.backtest            # unos 20 minutos en CPU
"""

import numpy as np
import pandas as pd

from credit_lab.features import derivar
from credit_lab.motor import CUOTA_MAX, ROI_EQUILIBRIO, TASAS, Motor, cuota, politica_lote, tasa_minima_lote
from credit_lab.negocio import utilidad_prestamo
from credit_lab.poblacion import RAIZ
from credit_lab.prepago import rendimiento

RUTA = RAIZ / "data" / "interim" / "backtest_politica.parquet"
RUTA_MUESTRA = RAIZ / "data" / "interim" / "backtest_muestra.parquet"  # las solicitudes de la muestra
N, SEMILLA = 10_000, 0
BLOQUE = 500
EFECTOS = [0.0, 0.005, 0.01, 0.02]  # PD que sube por punto de sobreprecio (supuesto del estrés)
COLUMNAS = ["loan_amnt", "int_rate", "annual_inc", "sub_grade", "home_ownership", "addr_state", "malo",
            "issue_d", "total_rec_int"]


def politica_anterior(p: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Decisión 020: aprobar → contraoferta por tasa (sin tope) → contraoferta por monto → rechazar."""
    t = p["tasa_sugerida"].to_numpy()
    ingreso, pedido = df["annual_inc"].to_numpy(float), df["loan_amnt"].to_numpy(float)
    por_tasa = ~np.isnan(t) & (np.nan_to_num(t, nan=TASAS.max()) <= TASAS.max()) & (
        cuota(pedido, np.nan_to_num(t, nan=TASAS.max())) / (ingreso / 12) <= CUOTA_MAX)
    por_monto = (p["monto_maximo"].to_numpy() > 0) & (p["monto_maximo"].to_numpy() < pedido)
    aprueba = p["regla"].eq("1").to_numpy()
    return pd.Series(np.select([aprueba, por_tasa, por_monto], ["1", "2a", "2b"], "3"), index=df.index)


def _por_bloques(f, df: pd.DataFrame, tamano: int = BLOQUE):
    """Aplica f por bloques: la grilla monto × tasa multiplica las filas y no cabe en memoria de una vez."""
    return pd.concat([f(df.iloc[i:i + tamano]) for i in range(0, len(df), tamano)])


def calcular(n: int = N, semilla: int = SEMILLA, etapas: int | None = None) -> pd.DataFrame:
    """Calcula por etapas y guarda después de cada una: si el proceso se corta, la siguiente corrida retoma
    donde quedó. `etapas` limita cuántas etapas nuevas se corren en esta llamada (para corridas cortas)."""
    from credit_lab.evals import cargar_muestra

    if RUTA.exists():
        out = pd.read_parquet(RUTA)
    else:
        s2 = cargar_muestra()
        df = s2.sample(min(n, len(s2)), random_state=semilla)
        del s2  # solo se necesita la muestra: libera la población completa
        out = df.reindex(columns=[c for c in COLUMNAS if c in df]).copy()
        out["posicion"] = df.index
        out["antig_credito_meses"] = derivar(df)["antig_credito_meses"].to_numpy()  # proxy de edad (issue #39)
        out["utilidad_real"] = utilidad_prestamo(df).to_numpy()
        out["roi_real"] = rendimiento(df).to_numpy()
        RUTA.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RUTA_MUESTRA)
        out.to_parquet(RUTA)
    df = pd.read_parquet(RUTA_MUESTRA)
    hechas = 0
    if "pol_regla" not in out:
        motor = Motor()
        base = _por_bloques(lambda x: politica_lote(motor, x), df)
        out = out.join(base.add_prefix("pol_"))
        out["pol_regla_anterior"] = politica_anterior(base, df)
        out.to_parquet(RUTA)
        print("política base lista", flush=True)
        hechas += 1
    for e in EFECTOS:
        clave = f"{e * 100:g}".replace(".", "_")
        m = Motor(estres=True, efecto_estres=e)
        pasos = {"equilibrio": lambda x: pd.Series(tasa_minima_lote(m, x, ROI_EQUILIBRIO), index=x.index),
                 "regla": lambda x: politica_lote(m, x)["regla"]}
        for nombre, f in pasos.items():  # cada subetapa se guarda aparte: así cabe en una corrida corta
            col = f"estres{clave}_{nombre}"
            if col in out or (etapas is not None and hechas >= etapas):
                continue
            out[col] = _por_bloques(f, df).to_numpy()
            out.to_parquet(RUTA)
            print(f"estrés {e:.3f} · {nombre} listo", flush=True)
            hechas += 1
    return out


def completo(b: pd.DataFrame) -> bool:
    return "pol_regla" in b and all(f"estres{f'{e * 100:g}'.replace('.', '_')}_regla" in b for e in EFECTOS)


def cargar() -> pd.DataFrame:
    return pd.read_parquet(RUTA)


def cambio_politica(b: pd.DataFrame) -> pd.DataFrame:
    """Issue #36: cómo cambia cada solicitud de la política anterior a la vigente, con su resultado real."""
    t = b.groupby(["pol_regla_anterior", "pol_regla"]).agg(
        solicitudes=("malo", "size"), mora_real=("malo", "mean"), roi_real=("roi_real", "mean"))
    t["%"] = t["solicitudes"] / len(b)
    return t


def sensibilidad_estres(b: pd.DataFrame) -> pd.DataFrame:
    """Issue #42: con cada efecto supuesto, cuántos quedan sin tasa viable, cómo rindieron y cuántas
    decisiones cambian frente a la política sin estrés."""
    filas = []
    for e in EFECTOS:
        clave = f"{e * 100:g}".replace(".", "_")
        sin = b[f"estres{clave}_equilibrio"].isna()
        filas.append({"efecto (pts de PD por pt de sobreprecio)": e * 100,
                      "sin tasa viable": sin.mean(), "mora real sin tasa viable": b.loc[sin, "malo"].mean(),
                      "ROI real sin tasa viable": b.loc[sin, "roi_real"].mean(),
                      "decisiones que cambian": (b[f"estres{clave}_regla"] != b["pol_regla"]).mean()})
    return pd.DataFrame(filas).set_index("efecto (pts de PD por pt de sobreprecio)")


def carga_analista(b: pd.DataFrame, minutos=(5, 8), por_cada: int = 10_000) -> dict:
    """Issue #38: cuántas solicitudes irían a firma (todo lo que no es aprobación de la política) y cuántas
    horas de analista serían. No incluye los desacuerdos del comité, que solo se miden con el agente."""
    a_firma = (b["pol_decision"] != "aprobar").mean()
    casos = a_firma * por_cada
    return {"a_firma": a_firma, "casos_por_10000": casos,
            "horas_por_10000": (casos * minutos[0] / 60, casos * minutos[1] / 60)}


def main() -> None:
    import sys

    b = calcular(etapas=int(sys.argv[1]) if len(sys.argv) > 1 else None)
    if not completo(b):
        print("Faltan etapas: vuelva a correr el comando para continuar.")
        return
    pd.set_option("display.float_format", "{:,.3f}".format)
    print(b["pol_decision"].value_counts(normalize=True))
    print(cambio_politica(b))
    print(sensibilidad_estres(b))
    print(carga_analista(b))


if __name__ == "__main__":
    main()

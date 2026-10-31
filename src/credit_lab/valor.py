"""¿Cuánto vale en dinero la mejora del AUC? (decisión 021)

Compara dos reglas de rechazo en 2015 (fuera de tiempo), con la misma tasa de
rechazo: ordenar por la PD del campeón u ordenar por la tasa de Lending Club
(su score). Para lo aprobado se mide la utilidad real (lo cobrado menos lo
prestado y la comisión de cobranza) y la mora real, escaladas a 10.000
solicitudes. El IC 95% de la diferencia sale de un bootstrap por préstamo.

Equidad: los datos públicos no traen atributos protegidos (sexo, edad, raza), así
que se revisan proxies (vivienda y región): tasa de rechazo y calibración
(PD media frente a mora real) por grupo, con el corte de rechazo del 20%.

Uso:
    python -m credit_lab.valor     # imprime las tablas y guarda docs/img/valor_campeon.png
"""

import numpy as np
import pandas as pd

from credit_lab.negocio import utilidad_prestamo
from credit_lab.poblacion import RAIZ, cargar_poblacion

POR_CADA = 10_000
RECHAZOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
REPLICAS = 500
SEMILLA = 0


def aprobados(score, tasa_rechazo: float) -> np.ndarray:
    """Aprueba a todos menos al tasa_rechazo con el score más alto (más riesgoso)."""
    return np.asarray(score) <= np.quantile(score, 1 - tasa_rechazo)


def comparar(utilidad, malo, pd_campeon, tasa_lc, rechazos=RECHAZOS, replicas=REPLICAS, semilla=SEMILLA) -> pd.DataFrame:
    """Utilidad y mora por cada 10.000 solicitudes con cada regla, y la diferencia con IC 95%."""
    u, y = np.asarray(utilidad, dtype=float), np.asarray(malo, dtype=float)
    n, rng = len(u), np.random.default_rng(semilla)
    muestras = rng.integers(0, n, size=(replicas, n))
    filas = []
    for r in rechazos:
        a_c, a_l = aprobados(pd_campeon, r), aprobados(tasa_lc, r)
        dif = (u * (a_c.astype(float) - a_l)) * POR_CADA / n  # utilidad extra por préstamo, escalada
        boot = dif[muestras].sum(axis=1)
        filas.append({
            "rechazo": r,
            "utilidad_campeon": u[a_c].sum() * POR_CADA / n,
            "utilidad_lending_club": u[a_l].sum() * POR_CADA / n,
            "diferencia": dif.sum(),
            "ic_bajo": np.percentile(boot, 2.5), "ic_alto": np.percentile(boot, 97.5),
            "mora_campeon": y[a_c].mean(), "mora_lending_club": y[a_l].mean(),
        })
    return pd.DataFrame(filas).set_index("rechazo")


REGIONES = {
    "Noreste": "CT ME MA NH RI VT NJ NY PA",
    "Medio Oeste": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "Sur": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "Oeste": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
ESTADO_REGION = {e: r for r, estados in REGIONES.items() for e in estados.split()}


def por_grupo(pd_modelo, malo, grupo, tasa_rechazo: float = 0.20, minimo: int = 1000) -> pd.DataFrame:
    """Rechazo y calibración por grupo. Una brecha de calibración cercana a 0 indica que el grupo
    no queda castigado de más (o de menos) respecto de su riesgo real."""
    d = pd.DataFrame({"pd": np.asarray(pd_modelo, dtype=float), "malo": np.asarray(malo, dtype=float),
                      "grupo": np.asarray(grupo)})
    d["rechazado"] = ~aprobados(d["pd"], tasa_rechazo)
    t = d.groupby("grupo").agg(prestamos=("pd", "size"), rechazo=("rechazado", "mean"),
                               pd_media=("pd", "mean"), mora_real=("malo", "mean"))
    t["brecha_calibracion"] = t["pd_media"] - t["mora_real"]
    return t[t["prestamos"] >= minimo]


def grafico(tabla: pd.DataFrame, ruta) -> None:
    import matplotlib.pyplot as plt

    azul, azul_claro, tinta, tinta_2, grilla = "#2a78d6", "#cde2fb", "#0b0b0b", "#52514e", "#e5e4df"
    x = tabla.index * 100
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.fill_between(x, tabla["ic_bajo"] / 1000, tabla["ic_alto"] / 1000, color=azul_claro, linewidth=0)
    ax.plot(x, tabla["diferencia"] / 1000, color=azul, linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.axhline(0, color=tinta_2, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(grilla)
    ax.grid(axis="y", color=grilla, linewidth=0.6)
    ax.tick_params(colors=tinta_2)
    ax.set_xticks(x, [f"{v:.0f}%" for v in x])
    ax.set_xlabel("Solicitudes rechazadas (las más riesgosas según cada score)", color=tinta_2)
    ax.set_ylabel("Miles de USD", color=tinta_2)
    ax.set_title("Utilidad extra del campeón frente al score de Lending Club\npor cada 10.000 solicitudes · 2015 fuera de tiempo · IC 95%",
                 loc="left", fontsize=10, fontweight="bold", color=tinta)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor="white")


def main() -> None:
    df = cargar_poblacion()
    oot = df[df["muestra"].eq("oot")]
    pred = pd.read_parquet(RAIZ / "modelos" / "predicciones_oot.parquet").loc[oot.index]
    tabla = comparar(utilidad_prestamo(oot), oot["malo"], pred["pd_lgbm"], oot["int_rate"])
    pd.set_option("display.float_format", "{:,.3f}".format)
    print(tabla)
    grafico(tabla, RAIZ / "docs" / "img" / "valor_campeon.png")

    import joblib

    from credit_lab.campeon import recalibrar

    pd_campeon = recalibrar(pred["pd_lgbm"], joblib.load(RAIZ / "modelos" / "campeon.joblib")["desplazamiento"])
    print(por_grupo(pd_campeon, oot["malo"], oot["home_ownership"]))
    print(por_grupo(pd_campeon, oot["malo"], oot["addr_state"].map(ESTADO_REGION).fillna("Otro")))


if __name__ == "__main__":
    main()

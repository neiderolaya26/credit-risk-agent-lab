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


# ---------- P&L de la política real, en neto (decisión 023, issue #35) ----------

# SUPUESTOS: valores dummy ilustrativos, no datos; el repo no tiene costos reales. Se reportan con una tabla de sensibilidad.
COSTOS = {
    "fondeo_anual": 0.03,       # costo del dinero prestado, sobre el saldo medio
    "operativo_prestamo": 100,  # USD por préstamo colocado (originación, servicio, cobranza)
    "capital": 0.10,            # capital que se reserva por cada USD prestado
    "costo_capital_anual": 0.12,
    "vida_media_anos": 1.5,     # años de saldo medio de un préstamo a 36 meses con prepago y castigos
}
ACEPTACION = [0.0, 0.5, 1.0]  # % de contraofertas que el cliente acepta: no hay datos (issue #36)
CASTIGO_SELECCION = 0.01      # PD que sube por punto de tasa extra en quien acepta (supuesto del estrés)
SEVERIDAD = 0.6               # pérdida sobre el monto cuando un préstamo extra cae en mora (supuesto)


def _utilidad_politica(b: pd.DataFrame, aceptacion: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Utilidad bruta, monto colocado y préstamos por solicitud con la política real.
    - aprobar: el préstamo tal cual (resultado real);
    - contraoferta por monto (2b): el rendimiento real del préstamo aplicado al monto ofrecido;
    - contraoferta por tasa (2a): los intereses reales escalados por la tasa nueva, menos el castigo por
      selección adversa (PD +1 pt por punto extra, con la severidad supuesta);
    - las contraofertas se aceptan con probabilidad `aceptacion` (en esperanza); rechazar: nada."""
    monto_lc, tasa_lc = b["loan_amnt"].to_numpy(float), b["int_rate"].to_numpy(float)
    u_lc, roi = b["utilidad_real"].to_numpy(float), b["roi_real"].to_numpy(float)
    regla, monto_of, tasa_of = b["pol_regla"].to_numpy(), b["pol_monto"].to_numpy(float), b["pol_tasa"].to_numpy(float)
    delta = np.maximum(tasa_of - tasa_lc, 0)
    extra_int = b["total_rec_int"].to_numpy(float) * np.divide(delta, tasa_lc, out=np.zeros_like(delta), where=tasa_lc > 0)
    castigo = CASTIGO_SELECCION * delta * SEVERIDAD * monto_lc
    u = np.select([regla == "1", regla == "2b", regla == "2a"],
                  [u_lc, roi * monto_of, u_lc + extra_int - castigo], 0.0)
    peso = np.select([regla == "1", np.isin(regla, ["2a", "2b"])], [1.0, aceptacion], 0.0)
    monto = np.select([regla == "1", regla == "2b", regla == "2a"], [monto_lc, monto_of, monto_lc], 0.0)
    return u * peso, monto * peso, peso


def _neto(u, monto, prestamos, costos: dict) -> np.ndarray:
    c = costos
    costo = monto * c["vida_media_anos"] * (c["fondeo_anual"] + c["capital"] * c["costo_capital_anual"])
    return u - costo - prestamos * c["operativo_prestamo"]


def escenarios(b: pd.DataFrame, costos: dict = COSTOS, replicas: int = REPLICAS, semilla: int = SEMILLA) -> pd.DataFrame:
    """P&L por cada 10.000 solicitudes de jul–dic 2015: Lending Club aprobando todo, la política real con
    0/50/100% de aceptación de contraofertas y un corte por subgrado que rechaza el mismo % que la política
    con 0% de aceptación. IC 95% por bootstrap de solicitudes."""
    n, rng = len(b), np.random.default_rng(semilla)
    monto_lc = b["loan_amnt"].to_numpy(float)
    casos = {"Lending Club aprueba todo": (b["utilidad_real"].to_numpy(float), monto_lc, np.ones(n))}
    for q in ACEPTACION:
        casos[f"Política real · acepta {q:.0%} de contraofertas"] = _utilidad_politica(b, q)
    rechazo = (b["pol_regla"] != "1").mean()
    orden = b["sub_grade"].rank(method="first")  # los subgrados más riesgosos quedan al final
    aprueba = (orden <= n * (1 - rechazo)).to_numpy()
    casos[f"Corte por subgrado (rechaza {rechazo:.0%})"] = (b["utilidad_real"].to_numpy(float) * aprueba,
                                                           monto_lc * aprueba, aprueba.astype(float))
    muestras = rng.integers(0, n, size=(replicas, n))
    filas = {}
    for nombre, (u, m, k) in casos.items():
        neto = _neto(u, m, k, costos)
        boot = neto[muestras].sum(axis=1) * POR_CADA / n
        filas[nombre] = {"préstamos": k.sum() * POR_CADA / n, "monto colocado": m.sum() * POR_CADA / n,
                         "utilidad bruta": u.sum() * POR_CADA / n, "utilidad neta": neto.sum() * POR_CADA / n,
                         "IC bajo": np.percentile(boot, 2.5), "IC alto": np.percentile(boot, 97.5),
                         "ROI neto sobre lo colocado": neto.sum() / m.sum() if m.sum() else np.nan}
    return pd.DataFrame(filas).T


def sensibilidad_costos(b: pd.DataFrame, fondeos=(0.02, 0.03, 0.05), operativos=(50, 100, 200)) -> pd.DataFrame:
    """Utilidad neta por 10.000 solicitudes de LC y de la política real (50% de aceptación), según costos."""
    filas = []
    for f in fondeos:
        for o in operativos:
            e = escenarios(b, COSTOS | {"fondeo_anual": f, "operativo_prestamo": o}, replicas=1)
            filas.append({"fondeo anual": f, "USD operativos por préstamo": o,
                          "LC aprueba todo": e.iloc[0]["utilidad neta"],
                          "política (50% acepta)": e.loc["Política real · acepta 50% de contraofertas", "utilidad neta"]})
    return pd.DataFrame(filas)


def equidad_precio(b: pd.DataFrame, grupo: str, minimo: int = 200) -> pd.DataFrame:
    """Issue #39: impacto dispar en rechazo y en subida de tasa (contraoferta 2a) por grupo, con la política real.
    La razón de impacto es la tasa del grupo dividida por la del grupo con menos (umbral fijado antes: 1,25)."""
    d = b.assign(rechazo=b["pol_regla"].eq("3"), sube_tasa=b["pol_regla"].eq("2a"),
                 subida=np.where(b["pol_regla"].eq("2a"), b["pol_tasa"] - b["int_rate"], 0.0))
    t = d.groupby(grupo).agg(solicitudes=("malo", "size"), rechazo=("rechazo", "mean"), sube_tasa=("sube_tasa", "mean"),
                             subida_media_pts=("subida", "mean"), mora_real=("malo", "mean"))
    t = t[t["solicitudes"] >= minimo]
    for c in ("rechazo", "sube_tasa"):
        base = t[c][t[c] > 0].min() if (t[c] > 0).any() else np.nan
        t[f"razon_{c}"] = t[c] / base
    return t


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
    # issue #24: el desplazamiento se ajustó con ene–jun 2015, así que la calibración se mide solo en jul–dic
    s2 = (oot["issue_d"] >= "2015-07-01").to_numpy()
    print(por_grupo(np.asarray(pd_campeon)[s2], oot["malo"][s2], oot["home_ownership"][s2]))
    print(por_grupo(np.asarray(pd_campeon)[s2], oot["malo"][s2], oot["addr_state"][s2].map(ESTADO_REGION).fillna("Otro")))

    from credit_lab import backtest

    if backtest.RUTA.exists():  # P&L de la política real y equidad en precio (decisión 023)
        b = backtest.cargar()
        if "total_rec_int" not in b:
            jul_dic = df[df["issue_d"] >= "2015-07-01"].reset_index(drop=True)
            b["total_rec_int"] = jul_dic.loc[b["posicion"], "total_rec_int"].to_numpy()
        b["region"] = b["addr_state"].map(ESTADO_REGION).fillna("Otro")
        print(escenarios(b))
        print(sensibilidad_costos(b))
        for g in ("home_ownership", "region"):
            print(equidad_precio(b, g))


if __name__ == "__main__":
    main()

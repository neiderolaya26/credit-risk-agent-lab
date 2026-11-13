"""Publica los modelos del motor en Hugging Face y arma los ejemplos de la demo.

    python -m credit_lab.publicar_hf ejemplos   # demo/ejemplos.csv (200 solicitudes de jul–dic 2015)
    python -m credit_lab.publicar_hf usuario    # la cuenta del token (HF_TOKEN en .env)
    python -m credit_lab.publicar_hf publicar   # sube huggingface/modelo y los 4 modelos al Hub

La demo (demo/streamlit_app.py) corre en Streamlit Community Cloud y descarga los modelos del Hub.
Los Spaces de Gradio en Hugging Face piden la suscripción PRO desde 2026.

Ejemplos: muestra del backtest estratificada por la decisión de la política, para que se vean las tres
salidas. Se quitan los identificadores y todo lo que se conoce después de originar el préstamo; se
conserva `malo` para mostrar el resultado real.
"""

import argparse
import os
import shutil
import sys

import pandas as pd

from credit_lab.poblacion import RAIZ

REPO_MODELO = "neiderolaya26/credit-risk-lending-club"
ARCHIVOS_MODELO = ["campeon.joblib", "prepago_c_supervivencia.joblib",
                   "severidad_b_economica.joblib", "referencia_tasa.joblib"]
# identificadores y variables posteriores a la originación: no van en los ejemplos
FUERA = ["id", "loan_status", "last_pymnt_d", "hardship_flag", "debt_settlement_flag", "total_pymnt",
         "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee", "out_prncp",
         "last_fico_range_high", "muestra", "anio"]
CUOTAS = {"aprobar": 120, "contraoferta": 50, "rechazar": 30}
SEMILLA = 26


def muestra_ejemplos() -> pd.DataFrame:
    """Las solicitudes de la demo con el índice y los tipos del parquet del backtest."""
    muestra = pd.read_parquet(RAIZ / "data" / "interim" / "backtest_muestra.parquet")
    decision = pd.read_parquet(RAIZ / "data" / "interim" / "backtest_politica.parquet")["pol_decision"]
    partes = [muestra.loc[decision[decision == d].index].sample(min(n, (decision == d).sum()), random_state=SEMILLA)
              for d, n in CUOTAS.items()]
    return pd.concat(partes).sample(frac=1, random_state=SEMILLA)


def ejemplos() -> None:
    df = muestra_ejemplos().drop(columns=FUERA, errors="ignore").reset_index(drop=True)
    ruta = RAIZ / "demo" / "ejemplos.csv"
    df.to_csv(ruta, index=False)
    print(f"{ruta}: {len(df)} solicitudes, {df['malo'].mean():.0%} en mora")


def _token() -> str:
    from credit_lab.agente import cargar_env

    cargar_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Falta HF_TOKEN en .env (token de escritura de https://huggingface.co/settings/tokens).")
    return token


def usuario() -> str:
    from huggingface_hub import whoami

    return whoami(token=_token())["name"]


def publicar() -> None:
    from huggingface_hub import HfApi

    carpeta = RAIZ / "data" / "hf" / "modelo"
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)
    shutil.copy2(RAIZ / "huggingface" / "modelo" / "README.md", carpeta / "README.md")
    for nombre in ARCHIVOS_MODELO:
        shutil.copy2(RAIZ / "modelos" / nombre, carpeta / nombre)

    api = HfApi(token=_token())
    api.create_repo(REPO_MODELO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO_MODELO, repo_type="model", folder_path=carpeta,
                      commit_message="Modelos del motor de credit-risk-agent-lab")
    print(f"Modelo: https://huggingface.co/{REPO_MODELO}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["ejemplos", "usuario", "publicar"])
    a = p.parse_args()
    if a.accion == "usuario":
        print(usuario())
    else:
        {"ejemplos": ejemplos, "publicar": publicar}[a.accion]()

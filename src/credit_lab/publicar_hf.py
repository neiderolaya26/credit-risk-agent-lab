"""Arma y publica en Hugging Face el Space del motor y el repo de modelos.

    python -m credit_lab.publicar_hf armar      # data/hf/space y data/hf/modelo (no publica nada)
    python -m credit_lab.publicar_hf probar     # copia los modelos al Space armado y lo abre en local
    python -m credit_lab.publicar_hf publicar   # sube los dos repos al Hub (pide HF_TOKEN en .env)

Ejemplos del Space: 200 solicitudes de jul–dic 2015 (fuera de tiempo) de la muestra del backtest,
estratificadas por la decisión de la política para que se vean las tres salidas. Se quitan los
identificadores y todo lo que se conoce después de originar el préstamo; se conserva `malo` para
mostrar el resultado real.
"""

import argparse
import os
import shutil
import sys

import pandas as pd

from credit_lab.poblacion import RAIZ

FUENTE = RAIZ / "huggingface"
DESTINO = RAIZ / "data" / "hf"
USUARIO_PREVISTO = "neiderolaya26"
SPACE, MODELO = "motor-credito", "credit-risk-lending-club"
ARCHIVOS_MODELO = ["campeon.joblib", "prepago_c_supervivencia.joblib",
                   "severidad_b_economica.joblib", "referencia_tasa.joblib"]
# identificadores y variables posteriores a la originación: no van en los ejemplos
FUERA = ["id", "loan_status", "last_pymnt_d", "hardship_flag", "debt_settlement_flag", "total_pymnt",
         "total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee", "out_prncp",
         "last_fico_range_high", "muestra", "anio"]
CUOTAS = {"aprobar": 120, "contraoferta": 50, "rechazar": 30}
IGNORAR = ["**/__pycache__/**", "**/*.pyc", "modelos/**", ".gradio/**", "flagged/**"]


def _ejemplos(semilla: int = 26) -> pd.DataFrame:
    muestra = pd.read_parquet(RAIZ / "data" / "interim" / "backtest_muestra.parquet")
    decision = pd.read_parquet(RAIZ / "data" / "interim" / "backtest_politica.parquet")["pol_decision"]
    partes = [muestra.loc[decision[decision == d].index].sample(min(n, (decision == d).sum()), random_state=semilla)
              for d, n in CUOTAS.items()]
    df = pd.concat(partes).sample(frac=1, random_state=semilla).drop(columns=FUERA, errors="ignore")
    return df.reset_index(drop=True)


def _usuario_real(texto: str, usuario: str) -> str:
    """Cambia solo los ids de los repos del Hub; los enlaces a GitHub quedan igual."""
    for repo in (SPACE, MODELO):
        texto = texto.replace(f"{USUARIO_PREVISTO}/{repo}", f"{usuario}/{repo}")
    return texto


def armar(usuario: str = USUARIO_PREVISTO) -> None:
    space, modelo = DESTINO / "space", DESTINO / "modelo"
    for carpeta in (space, modelo):
        if carpeta.exists():
            shutil.rmtree(carpeta)
        carpeta.mkdir(parents=True)

    for nombre in ("app.py", "README.md", "requirements.txt"):
        texto = (FUENTE / "space" / nombre).read_text(encoding="utf-8")
        (space / nombre).write_text(_usuario_real(texto, usuario), encoding="utf-8")
    shutil.copytree(RAIZ / "src" / "credit_lab", space / "src" / "credit_lab",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    ej = _ejemplos()
    ej.to_csv(space / "ejemplos.csv", index=False)

    texto = (FUENTE / "modelo" / "README.md").read_text(encoding="utf-8")
    (modelo / "README.md").write_text(_usuario_real(texto, usuario), encoding="utf-8")
    for nombre in ARCHIVOS_MODELO:
        shutil.copy2(RAIZ / "modelos" / nombre, modelo / nombre)

    mb = sum(p.stat().st_size for p in modelo.iterdir()) / 1e6
    print(f"Space:  {space}  ({len(ej)} ejemplos, {ej['malo'].mean():.0%} en mora)")
    print(f"Modelo: {modelo}  ({len(ARCHIVOS_MODELO)} archivos, {mb:.1f} MB)")


def probar() -> None:
    space = DESTINO / "space"
    if not space.exists():
        armar()
    (space / "modelos").mkdir(exist_ok=True)
    for nombre in ARCHIVOS_MODELO:  # en el Hub los descarga app.py; aquí se copian para no depender de él
        shutil.copy2(RAIZ / "modelos" / nombre, space / "modelos" / nombre)
    os.chdir(space)
    os.execv(sys.executable, [sys.executable, str(space / "app.py")])


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

    token = _token()
    api = HfApi(token=token)
    quien = api.whoami()["name"]
    armar(quien)
    id_modelo, id_space = f"{quien}/{MODELO}", f"{quien}/{SPACE}"
    api.create_repo(id_modelo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=id_modelo, repo_type="model", folder_path=DESTINO / "modelo",
                      commit_message="Modelos del motor de credit-risk-agent-lab")
    api.create_repo(id_space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=id_space, repo_type="space", folder_path=DESTINO / "space",
                      ignore_patterns=IGNORAR, commit_message="Demo del motor de decisión de crédito")
    print(f"Modelo: https://huggingface.co/{id_modelo}")
    print(f"Space:  https://huggingface.co/spaces/{id_space}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["armar", "probar", "usuario", "publicar"])
    a = p.parse_args()
    if a.accion == "usuario":
        print(usuario())
    else:
        {"armar": armar, "probar": probar, "publicar": publicar}[a.accion]()

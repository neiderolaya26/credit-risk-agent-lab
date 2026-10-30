"""Convierte el CSV comprimido de Lending Club a Parquet, por bloques.

El archivo pesa ~1,7 GB descomprimido; se lee en bloques y solo con las
columnas útiles para no agotar la memoria.

Uso:
    python -m credit_lab.ingest
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[2]
ORIGEN = RAIZ / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
DESTINO = RAIZ / "data" / "interim" / "accepted.parquet"

# Texto: se guardan tal cual y se limpian después.
COLS_TEXTO = [
    "id", "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "issue_d", "loan_status", "purpose", "addr_state",
    "earliest_cr_line", "initial_list_status", "application_type",
    "last_pymnt_d", "hardship_flag", "debt_settlement_flag",
]

# Numéricas: variables de la solicitud y del resultado del préstamo.
COLS_NUM = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "acc_now_delinq",
    "tot_cur_bal", "total_rev_hi_lim", "bc_util", "num_actv_rev_tl",
    "percent_bc_gt_75", "avg_cur_bal", "num_tl_90g_dpd_24m", "pct_tl_nvr_dlq",
    "mo_sin_old_rev_tl_op", "total_bc_limit",
    # Resultado (no se usan como features: fuga de información)
    "total_pymnt", "recoveries", "out_prncp", "last_fico_range_high",
]

ESQUEMA = pa.schema(
    [(c, pa.string()) for c in COLS_TEXTO] + [(c, pa.float32()) for c in COLS_NUM]
)


def convertir(tam_bloque: int = 200_000) -> Path:
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    escritor = pq.ParquetWriter(DESTINO, ESQUEMA)
    filas = 0
    try:
        bloques = pd.read_csv(
            ORIGEN,
            usecols=COLS_TEXTO + COLS_NUM,
            dtype={c: "string" for c in COLS_TEXTO},
            chunksize=tam_bloque,
            low_memory=False,
        )
        for bloque in bloques:
            for c in COLS_NUM:
                bloque[c] = pd.to_numeric(bloque[c], errors="coerce").astype("float32")
            # El CSV trae filas de resumen al final sin monto de préstamo.
            bloque = bloque.dropna(subset=["loan_amnt"])
            tabla = pa.Table.from_pandas(bloque[ESQUEMA.names], schema=ESQUEMA, preserve_index=False)
            escritor.write_table(tabla)
            filas += len(bloque)
            print(f"  {filas:,} filas", flush=True)
    finally:
        escritor.close()
    print(f"Listo: {DESTINO} ({filas:,} filas)")
    return DESTINO


if __name__ == "__main__":
    convertir()

"""Carga de los datos crudos de accidentes de tráfico de Madrid (2019-2025)."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def cargar_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Lee todos los CSV `accidentes_trafico_AAAA.csv` y los concatena.

    Devuelve un único DataFrame con las columnas originales (sin limpiar)
    más la columna `año`, tomada del nombre del fichero. Cada fila es una
    persona implicada, no un accidente.
    """
    ficheros = sorted(raw_dir.glob("accidentes_trafico_*.csv"))
    if not ficheros:
        raise FileNotFoundError(f"No se encontraron CSV en {raw_dir}")

    dfs = []
    for fichero in ficheros:
        df = pd.read_csv(fichero, sep=";", encoding="utf-8-sig", dtype=str)
        df["año"] = int(fichero.stem.rsplit("_", 1)[-1])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    datos = cargar_raw()
    print(datos.shape)
    print(datos["año"].value_counts().sort_index())

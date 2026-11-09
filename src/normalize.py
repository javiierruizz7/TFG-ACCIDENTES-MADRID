"""Normalización de nulos y tipos de los datos crudos de accidentes.

Va después de `load_data.cargar_raw()`. No quita filas ni inventa valores: solo unifica
los distintos "sin dato" a NaN y convierte cada columna a su tipo.
"""

import pandas as pd

# Con dtype=str, la cadena vacía ya llega como NaN; estos dos llegan como texto.
VALORES_NULOS = ["NULL", "#¡VALOR!"]

COLS_COORD = ["coordenada_x_utm", "coordenada_y_utm"]
COLS_ENTERO = ["cod_distrito", "cod_lesividad"]
COLS_CATEGORIA = [
    "distrito",
    "tipo_accidente",
    "estado_meteorológico",
    "tipo_vehiculo",
    "tipo_persona",
    "rango_edad",
    "sexo",
    "lesividad",
    "positiva_alcohol",
    "positiva_droga",
]
# `localizacion` y `numero` se dejan como texto (`numero` trae valores tipo "+00500E").


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con los nulos unificados a NaN y los tipos convertidos.

    Orden: primero los nulos y después los tipos, porque convertir antes haría fallar
    valores como `#¡VALOR!`. Al final comprueba que la conversión no ha
    creado ni borrado ningún nulo; si no coinciden, lanza ValueError.
    """
    out = df.replace({valor: pd.NA for valor in VALORES_NULOS})
    nulos_antes = out.isna().sum()

    out["fecha"] = pd.to_datetime(out["fecha"], format="%d/%m/%Y")
    out["hora"] = pd.to_timedelta(out["hora"])
    for col in COLS_COORD:
        out[col] = pd.to_numeric(out[col])
    for col in COLS_ENTERO:
        out[col] = pd.to_numeric(out[col]).astype("Int16")
    for col in COLS_CATEGORIA:
        out[col] = out[col].astype("category")

    nulos_despues = out.isna().sum()
    distintos = nulos_antes[nulos_antes != nulos_despues]
    if not distintos.empty:
        raise ValueError(f"La conversión de tipos cambió los nulos en: {list(distintos.index)}")
    return out

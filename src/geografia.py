"""Pasa a un accidente por fila y lo cruza con el contorno de Madrid.

Va después de `normalize.normalizar`. No descarta ningún accidente: solo añade columnas
que dicen si tiene coordenadas, si cae dentro del municipio y en qué distrito.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TERMINO = ROOT / "data" / "external" / "termino_municipal" / "TERMINO_MUNICIPAL.shp"
DISTRITOS = ROOT / "data" / "external" / "distritos" / "DISTRITOS.shp"
CRS = "EPSG:25830"

# Estas columnas deben valer lo mismo en todas las filas (personas) de un expediente.
COLS_EXPEDIENTE = [
    "fecha",
    "hora",
    "coordenada_x_utm",
    "coordenada_y_utm",
    "cod_distrito",
    "distrito",
    "localizacion",
    "año",
]


def a_expedientes(limpio: pd.DataFrame) -> pd.DataFrame:
    """Una fila por `num_expediente`. Lanza ValueError si algún expediente se contradice
    en fecha, hora o coordenadas entre sus filas (en los datos actuales no ocurre)."""
    grupos = limpio.groupby("num_expediente", observed=True)
    contradictorios = (grupos[["fecha", "hora", "coordenada_x_utm", "coordenada_y_utm"]]
                       .nunique(dropna=True) > 1).any(axis=1)
    if contradictorios.any():
        raise ValueError(f"{int(contradictorios.sum())} expedientes con datos contradictorios")
    out = grupos[COLS_EXPEDIENTE].first()
    out["n_personas"] = grupos.size()
    return out.reset_index()


def cruce_espacial(expedientes: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con cuatro columnas nuevas:

    - `sin_coord`: no tiene X/Y.
    - `dentro_municipio`: True/False; `<NA>` si no tiene coordenadas.
    - `distancia_limite_m`: 0 si está dentro; si está fuera, metros hasta el límite.
    - `distrito_geo`: código del distrito según la posición (`<NA>` si fuera o sin coordenadas).
    """
    municipio = gpd.read_file(TERMINO).to_crs(CRS).geometry.union_all()
    distritos = gpd.read_file(DISTRITOS).to_crs(CRS)
    distritos["COD_DIS"] = pd.to_numeric(distritos["COD_DIS"]).astype("Int16")

    out = expedientes.copy()
    out["sin_coord"] = out[["coordenada_x_utm", "coordenada_y_utm"]].isna().any(axis=1)
    con = out[~out["sin_coord"]]
    puntos = gpd.GeoDataFrame(
        index=con.index,
        geometry=gpd.points_from_xy(con["coordenada_x_utm"], con["coordenada_y_utm"]),
        crs=CRS,
    )

    dentro = puntos.within(municipio)
    out["dentro_municipio"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out.loc[con.index, "dentro_municipio"] = dentro

    out["distancia_limite_m"] = float("nan")
    out.loc[con.index, "distancia_limite_m"] = 0.0
    fuera = puntos[~dentro]
    out.loc[fuera.index, "distancia_limite_m"] = fuera.distance(municipio)

    cruce = puntos[dentro].sjoin(distritos[["COD_DIS", "geometry"]], predicate="within", how="left")
    cruce = cruce[~cruce.index.duplicated()]
    out["distrito_geo"] = pd.array([pd.NA] * len(out), dtype="Int16")
    out.loc[cruce.index, "distrito_geo"] = cruce["COD_DIS"]
    return out


def informe_perdidas(cruce: pd.DataFrame) -> pd.DataFrame:
    """Tabla por año: expedientes totales, excluidos por no tener coordenadas, los que se
    conservan aunque caigan fuera del límite (borde) y los que quedan en `accidents_clean`."""
    estado = cruce["sin_coord"].map({True: "sin_coordenadas", False: "con_coordenadas"})
    tabla = pd.DataFrame({"expedientes": cruce.groupby("año").size()})
    tabla["excluidos_sin_coordenadas"] = (
        cruce[cruce["sin_coord"]].groupby("año").size().reindex(tabla.index, fill_value=0)
    )
    fuera = cruce["dentro_municipio"].eq(False)
    tabla["conservados_fuera_de_limite"] = fuera.groupby(cruce["año"]).sum()
    tabla["en_accidents_clean"] = tabla["expedientes"] - tabla["excluidos_sin_coordenadas"]
    tabla.loc["total"] = tabla.sum()
    assert estado.notna().all()
    return tabla.astype(int)


def accidentes_limpios(cruce: pd.DataFrame) -> pd.DataFrame:
    """Expedientes con coordenadas: excluye los que no tienen y conserva los de borde
    (`dentro_municipio=False`, con su `distancia_limite_m`). Deja el distrito geográfico."""
    out = cruce[~cruce["sin_coord"]].drop(columns="sin_coord").reset_index(drop=True)
    out["dentro_municipio"] = out["dentro_municipio"].astype(bool)
    return out

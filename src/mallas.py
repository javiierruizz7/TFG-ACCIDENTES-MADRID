"""Mallas espaciales (1 km, 2 km y distritos) construidas solo a partir del contorno oficial.

No se mira ningún accidente para decidir qué celdas existen: así las zonas con cero accidentes
también están en el panel. Una celda cuenta como válida si solapa con el municipio (con área > 0),
aunque solo lo cubra en parte.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from geografia import CRS, DISTRITOS, TERMINO

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"


def _municipio() -> gpd.GeoDataFrame:
    return gpd.read_file(TERMINO).to_crs(CRS)


def crear_malla(lado_m: int) -> gpd.GeoDataFrame:
    """Malla regular de `lado_m` metros.

    Columnas: `cell_id` (0, 1, 2… ordenado por fila de sur a norte y luego de oeste a este),
    `x_min`, `y_min`, `lado_m`, `area_dentro_m2` y `frac_dentro` (fracción de la celda que cae
    dentro del municipio) y la geometría.
    """
    municipio = _municipio().geometry.union_all()
    minx, miny, maxx, maxy = municipio.bounds
    xs = np.arange(np.floor(minx / lado_m) * lado_m, maxx, lado_m)
    ys = np.arange(np.floor(miny / lado_m) * lado_m, maxy, lado_m)

    celdas = gpd.GeoDataFrame(
        {"x_min": np.tile(xs, len(ys)).astype(int), "y_min": np.repeat(ys, len(xs)).astype(int)},
        geometry=[box(x, y, x + lado_m, y + lado_m) for y in ys for x in xs],
        crs=CRS,
    )
    celdas["area_dentro_m2"] = celdas.geometry.intersection(municipio).area
    celdas = celdas[celdas["area_dentro_m2"] > 0].reset_index(drop=True)
    celdas["frac_dentro"] = celdas["area_dentro_m2"] / (lado_m**2)
    celdas["lado_m"] = lado_m
    celdas.insert(0, "cell_id", np.arange(len(celdas), dtype="int16"))
    return celdas


def malla_distritos() -> gpd.GeoDataFrame:
    """Los 21 distritos como malla: `cell_id` = código de distrito (1–21)."""
    d = gpd.read_file(DISTRITOS).to_crs(CRS)
    out = gpd.GeoDataFrame(
        {
            "cell_id": pd.to_numeric(d["COD_DIS"]).astype("int16"),
            "nombre": d["NOMBRE"],
            "area_dentro_m2": d.geometry.area,
        },
        geometry=d.geometry,
        crs=CRS,
    ).sort_values("cell_id")
    out["frac_dentro"] = 1.0
    return out.reset_index(drop=True)


def guardar_mallas(destino: Path = INTERIM) -> dict[str, Path]:
    """Guarda las tres mallas en GeoParquet (una por resolución) y devuelve las rutas."""
    rutas = {}
    for nombre, malla in [
        ("malla_1km", crear_malla(1000)),
        ("malla_2km", crear_malla(2000)),
        ("malla_distritos", malla_distritos()),
    ]:
        rutas[nombre] = destino / f"{nombre}.parquet"
        malla.to_parquet(rutas[nombre], index=False)
    return rutas


def _puntos(accidentes: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        index=accidentes.index,
        geometry=gpd.points_from_xy(accidentes["coordenada_x_utm"], accidentes["coordenada_y_utm"]),
        crs=CRS,
    )


def _celda_mas_cercana(puntos: gpd.GeoDataFrame, malla: gpd.GeoDataFrame) -> pd.Series:
    """`cell_id` de la celda más cercana a cada punto (empate: la de menor `cell_id`)."""
    cerca = puntos.sjoin_nearest(malla[["cell_id", "geometry"]], how="left")
    cerca = cerca.sort_values("cell_id")
    return cerca[~cerca.index.duplicated()]["cell_id"].sort_index()


def asignar_celda_cuadrada(accidentes: pd.DataFrame, malla: gpd.GeoDataFrame) -> pd.DataFrame:
    """Asigna cada accidente a su celda con intervalo semiabierto: `x_min <= x < x_min + lado`
    (y lo mismo en Y). Así un accidente justo sobre una línea va a la celda de su este/norte y
    nunca queda sin celda ni en dos. Devuelve `cell_id` y `por_cercania` (True solo si el punto
    cae fuera de todas las celdas y se asigna a la más cercana)."""
    lado = int(malla["lado_m"].iloc[0])
    clave = pd.DataFrame(
        {
            "x_min": (np.floor(accidentes["coordenada_x_utm"] / lado) * lado).astype(int),
            "y_min": (np.floor(accidentes["coordenada_y_utm"] / lado) * lado).astype(int),
        },
        index=accidentes.index,
    )
    cell = clave.merge(malla[["x_min", "y_min", "cell_id"]], on=["x_min", "y_min"], how="left")
    cell.index = accidentes.index
    sin = cell["cell_id"].isna()
    if sin.any():
        cell.loc[sin, "cell_id"] = _celda_mas_cercana(_puntos(accidentes[sin]), malla)
    return pd.DataFrame({"cell_id": cell["cell_id"].astype("int16"), "por_cercania": sin})


def asignar_distrito(accidentes: pd.DataFrame, malla: gpd.GeoDataFrame) -> pd.DataFrame:
    """Distrito de cada accidente por posición; si cae fuera de todos (borde del municipio),
    el distrito más cercano. Mismas columnas que `asignar_celda_cuadrada`."""
    puntos = _puntos(accidentes)
    dentro = puntos.sjoin(malla[["cell_id", "geometry"]], predicate="within", how="left")
    dentro = dentro[~dentro.index.duplicated()]["cell_id"].reindex(accidentes.index)
    sin = dentro.isna()
    if sin.any():
        dentro.loc[sin] = _celda_mas_cercana(puntos[sin], malla)
    return pd.DataFrame({"cell_id": dentro.astype("int16"), "por_cercania": sin})

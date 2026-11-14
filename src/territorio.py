"""Variables territoriales estáticas de cada zona (E3).

Salen solo de las geometrías oficiales (mallas y distritos): no dependen de ningún accidente ni
del tiempo, así que no pueden filtrar información del futuro.

- `coord_x`, `coord_y`: centroide de la parte de la zona dentro del municipio (EPSG:25830, en km).
- `area_km2`, `frac_dentro`: superficie dentro del municipio y fracción de la celda que cubre.
- `distrito`: código (1–21) del distrito con mayor solape con la celda; en la malla de distritos
  es el propio código.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from geografia import DISTRITOS, TERMINO
from mallas import INTERIM

VARIABLES_TERRITORIALES = ("coord_x", "coord_y", "area_km2", "frac_dentro", "distrito")
GRUPOS = {
    "posicion": ("coord_x", "coord_y"),
    "tamano": ("area_km2", "frac_dentro"),
    "distrito": ("distrito",),
}


def tabla_territorial(resolucion: str, carpeta: Path = INTERIM) -> pd.DataFrame:
    """Una fila por `cell_id` con las variables territoriales de esa resolución."""
    malla = gpd.read_parquet(carpeta / f"malla_{resolucion}.parquet")
    municipio = gpd.read_file(TERMINO).to_crs(malla.crs).geometry.union_all()
    dentro = malla.geometry.intersection(municipio)
    out = pd.DataFrame(
        {
            "cell_id": malla["cell_id"].to_numpy(),
            "coord_x": dentro.centroid.x.to_numpy() / 1000,
            "coord_y": dentro.centroid.y.to_numpy() / 1000,
            "area_km2": malla["area_dentro_m2"].to_numpy() / 1e6,
            "frac_dentro": malla["frac_dentro"].to_numpy(),
        }
    )
    if resolucion == "distritos":
        out["distrito"] = out["cell_id"].astype(int)
    else:
        dis = gpd.read_file(DISTRITOS).to_crs(malla.crs)
        codigos = pd.to_numeric(dis["COD_DIS"]).to_numpy()
        solape = pd.DataFrame(
            {
                c: dentro.intersection(g).area.to_numpy()
                for c, g in zip(codigos, dis.geometry, strict=True)
            }
        )
        out["distrito"] = solape.idxmax(axis=1).astype(int).to_numpy()
    return out


def con_territorio(panel: pd.DataFrame, resolucion: str) -> pd.DataFrame:
    """Añade las variables territoriales al panel (une por `cell_id`)."""
    t = tabla_territorial(resolucion)
    return panel.merge(t, on="cell_id", how="left", validate="many_to_one")

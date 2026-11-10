"""Panel celda x mes con ceros, sin geometría.

Una fila por cada combinación de celda y mes del estudio. `y` es el número de accidentes
(expedientes únicos) de esa celda en ese mes. La geometría vive aparte (mallas en GeoParquet)
y se une por `cell_id` solo cuando hace falta pintar mapas.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MESES = pd.date_range("2019-01-01", "2025-12-01", freq="MS")  # 84 meses completos


def construir_panel(
    asignacion: pd.DataFrame, cell_ids: pd.Series, col_celda: str, meses: pd.DatetimeIndex = MESES
) -> pd.DataFrame:
    """Producto cartesiano `cell_ids` x `meses` con el conteo `y` (0 donde no hay accidentes).

    `asignacion` trae una fila por accidente con `mes` (primer día del mes) y la columna de celda.
    Falla si algún accidente cae en una celda o mes que no está en el panel, así el total
    se conserva exactamente y no se pierde nada en silencio.
    """
    celdas = np.sort(cell_ids.unique())
    fuera = ~asignacion[col_celda].isin(celdas) | ~asignacion["mes"].isin(meses)
    if fuera.any():
        raise ValueError(f"{int(fuera.sum())} accidentes fuera de las celdas o meses del panel")

    indice = pd.MultiIndex.from_product([celdas, meses], names=["cell_id", "target_month"])
    conteo = asignacion.groupby([col_celda, "mes"]).size()
    conteo.index = conteo.index.set_names(["cell_id", "target_month"])
    panel = conteo.reindex(indice, fill_value=0).rename("y").reset_index()
    panel["cell_id"] = panel["cell_id"].astype("int16")
    panel["y"] = panel["y"].astype("int16")
    assert int(panel["y"].sum()) == len(asignacion)
    return panel

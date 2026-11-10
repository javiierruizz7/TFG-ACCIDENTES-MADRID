"""E0: baselines sin aprendizaje (persistencia, medias móviles y estacionalidad)."""

import numpy as np
import pandas as pd

from features import anadir_rezagos
from metricas import capture_at_k, devianza_poisson, mae, rmse

# nombre del baseline -> columna de rezago/media que usa como predicción
BASELINES = {
    "persistencia": "lag_1",
    "media_movil_3": "ma_3",
    "media_movil_6": "ma_6",
    "media_movil_12": "ma_12",
    "estacionalidad": "lag_12",
}


def evaluar_e0(
    panel: pd.DataFrame, meses_eval: pd.DatetimeIndex, latencia: int = 1
) -> pd.DataFrame:
    """Evalúa los baselines de E0 sobre `meses_eval` (p. ej. los 12 meses de 2024).

    No se entrena nada: la predicción de cada mes sale solo de meses anteriores
    (`anadir_rezagos`). Devuelve una fila por baseline con MAE, RMSE, devianza de Poisson
    (sobre todas las celdas y meses) y Capture@10 % (media de los meses con accidentes).
    """
    lags = (max(1, latencia), 12)
    datos = anadir_rezagos(panel, lags=lags, ventanas=(3, 6, 12), latencia=latencia)
    datos = datos[datos["target_month"].isin(meses_eval)]

    filas = []
    for nombre, columna in BASELINES.items():
        col = f"lag_{latencia}" if columna == "lag_1" else columna
        d = datos.dropna(subset=[col])
        capturas = [
            capture_at_k(m["y"], m[col], m["cell_id"]) for _, m in d.groupby("target_month")
        ]
        filas.append(
            {
                "baseline": nombre,
                "filas_evaluadas": len(d),
                "MAE": mae(d["y"], d[col]),
                "RMSE": rmse(d["y"], d[col]),
                "devianza_poisson": devianza_poisson(d["y"], d[col]),
                "capture@10%": float(np.nanmean(capturas)),
            }
        )
    return pd.DataFrame(filas)

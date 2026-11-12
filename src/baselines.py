"""E0: baselines sin aprendizaje (persistencia, medias móviles y estacionalidad).

Además de los cinco de las directrices hay un sexto, `media_12_estacional`: la media de 12 meses
multiplicada por el índice estacional del mes (p. ej. agosto ≈ 0,66). Sigue sin aprender nada,
pero sube el listón: corrige que la media de 12 meses predice casi lo mismo todos los meses.
"""

import numpy as np
import pandas as pd

from features import anadir_rezagos
from protocolo import AjustarPredecir, baseline_como_modelo, evaluar_rolling, resumir

# nombre del baseline -> columna de rezago/media que usa como predicción
BASELINES = {
    "persistencia": "lag_1",
    "media_movil_3": "ma_3",
    "media_movil_6": "ma_6",
    "media_movil_12": "ma_12",
    "estacionalidad": "lag_12",
}

# Años que no se usan para estimar la estacionalidad: 2020 está distorsionado por la COVID
# (marzo-mayo casi vacíos y enero-febrero inflados en proporción).
AÑOS_SIN_ESTACIONALIDAD = (2020,)


def indice_estacional(panel: pd.DataFrame, ultimo_conocido: pd.Timestamp) -> pd.Series:
    """Índice de cada mes del año (1–12): total del mes / media mensual de su año, promediado
    sobre los años **completos** con todos sus meses <= `ultimo_conocido` (sin 2020).

    Solo mira el pasado: es lo que se sabría al predecir el mes siguiente a `ultimo_conocido`.
    Si aún no hay ningún año completo, devuelve NaN para los 12 meses.
    """
    conocido = panel[panel["target_month"] <= ultimo_conocido]
    total = conocido.groupby("target_month")["y"].sum()
    tabla = pd.DataFrame({"total": total, "año": total.index.year, "mes": total.index.month})
    tabla = tabla[~tabla["año"].isin(AÑOS_SIN_ESTACIONALIDAD)]
    tabla = tabla.groupby("año").filter(lambda a: len(a) == 12)
    if tabla.empty:
        return pd.Series(np.nan, index=range(1, 13))
    tabla["indice"] = tabla["total"] / tabla.groupby("año")["total"].transform("mean")
    return tabla.groupby("mes")["indice"].mean().reindex(range(1, 13))


def media_12_estacional(panel: pd.DataFrame, latencia: int = 1) -> AjustarPredecir:
    """Baseline `ma_12 × índice estacional del mes`, con el índice estimado solo con el pasado."""

    def ajustar_predecir(entreno: pd.DataFrame, a_predecir: pd.DataFrame) -> np.ndarray:
        mes = a_predecir["target_month"].iloc[0]
        indice = indice_estacional(panel, mes - pd.DateOffset(months=latencia))
        factor = indice.loc[a_predecir["target_month"].dt.month].to_numpy()
        return a_predecir["ma_12"].to_numpy(dtype=float) * factor

    return ajustar_predecir


def modelos_e0(panel: pd.DataFrame, latencia: int = 1) -> dict[str, tuple[AjustarPredecir, str]]:
    """Los seis baselines como "modelos" del protocolo: nombre -> (función, columna que usa)."""
    modelos = {}
    for nombre, columna in BASELINES.items():
        col = f"lag_{latencia}" if columna == "lag_1" else columna
        modelos[nombre] = (baseline_como_modelo(col), col)
    modelos["media_12_estacional"] = (media_12_estacional(panel, latencia), "ma_12")
    return modelos


def predecir_e0(
    panel: pd.DataFrame, meses_eval: pd.DatetimeIndex, latencia: int = 1
) -> dict[str, pd.DataFrame]:
    """Predicciones de los seis baselines por el protocolo rolling-origin."""
    lags = (max(1, latencia), 12)
    datos = anadir_rezagos(panel, lags=lags, ventanas=(3, 6, 12), latencia=latencia)
    return {
        nombre: evaluar_rolling(datos, meses_eval, modelo, [col], latencia)
        for nombre, (modelo, col) in modelos_e0(panel, latencia).items()
    }


def evaluar_e0(
    panel: pd.DataFrame, meses_eval: pd.DatetimeIndex, latencia: int = 1
) -> pd.DataFrame:
    """Evalúa los baselines de E0 sobre `meses_eval` (p. ej. los 12 meses de 2024).

    Devuelve una fila por baseline con MAE, RMSE, devianza de Poisson (sobre todas las celdas
    y meses) y Capture@10 % (media de los meses con accidentes).
    """
    preds = predecir_e0(panel, meses_eval, latencia)
    filas = [{"baseline": n, **resumir(p)} for n, p in preds.items()]
    return pd.DataFrame(filas).rename(columns={"filas": "filas_evaluadas"})

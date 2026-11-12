"""Protocolo de evaluación: rolling-origin por meses completos, con reentrenamiento mensual.

Para cada mes m que se evalúa:
  1. se entrena con las filas cuya etiqueta ya se conoce al empezar m, es decir,
     `target_month <= m - latencia` (con latencia 1, hasta el mes anterior);
  2. se predice el mes m para todas las celdas a la vez;
  3. se pasa al mes siguiente y se vuelve a entrenar con un mes más.

Los rezagos de cada fila ya vienen calculados sin fuga (`features.anadir_rezagos`), así que
"actualizar las variables" es simplemente usar las columnas de la fila del mes m.
Las etiquetas de los meses >= m nunca llegan al modelo que predice m.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from metricas import capture_at_k, devianza_poisson, mae, rmse

MESES_VALIDACION = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
MESES_TEST = pd.date_range("2025-01-01", "2025-12-01", freq="MS")  # no usar hasta el final

# Recibe (entrenamiento, filas_a_predecir) y devuelve una predicción por fila a predecir.
AjustarPredecir = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

BANDAS = ["0", "1-2", ">=3"]


def banda_frecuencia(ma_12: pd.Series) -> pd.Series:
    """Banda de frecuencia histórica de cada fila según `ma_12` (media de los 12 meses anteriores).

    0 si `ma_12` = 0; "1-2" si está entre 0 y 3; ">=3" si es 3 o más. NaN sin historia.
    Nunca se usa el `y` del propio mes: la banda solo depende del pasado.
    """
    banda = pd.Series(pd.NA, index=ma_12.index, dtype="object")
    banda[ma_12 == 0] = "0"
    banda[(ma_12 > 0) & (ma_12 < 3)] = "1-2"
    banda[ma_12 >= 3] = ">=3"
    return pd.Series(
        pd.Categorical(banda, categories=BANDAS, ordered=True), index=ma_12.index, name="banda"
    )


def evaluar_rolling(
    datos: pd.DataFrame,
    meses_eval: pd.DatetimeIndex,
    ajustar_predecir: AjustarPredecir,
    columnas: list[str],
    latencia: int = 1,
) -> pd.DataFrame:
    """Aplica el protocolo y devuelve una fila por (celda, mes evaluado) con `y` e `yhat`.

    `datos` es el panel con rezagos (`anadir_rezagos`) y la columna `ma_12` para la banda.
    `columnas` son las variables que usa el modelo: las filas sin alguna de ellas (meses sin
    historia suficiente) no entran ni en el entrenamiento ni en la evaluación.
    """
    validas = datos.dropna(subset=columnas)
    resultados = []
    for mes in meses_eval:
        ultimo_conocido = mes - pd.DateOffset(months=latencia)
        entreno = validas[validas["target_month"] <= ultimo_conocido]
        a_predecir = validas[validas["target_month"] == mes]
        if a_predecir.empty:
            raise ValueError(f"No hay filas que predecir en {mes:%Y-%m}")
        # Comprobación explícita de que el modelo nunca ve la etiqueta del mes que predice.
        assert entreno.empty or entreno["target_month"].max() < mes

        yhat = np.asarray(ajustar_predecir(entreno, a_predecir), dtype=float)
        if len(yhat) != len(a_predecir):
            raise ValueError("El modelo devolvió un número de predicciones distinto")
        out = a_predecir[["cell_id", "target_month", "y"]].copy()
        out["yhat"] = yhat
        out["banda"] = banda_frecuencia(a_predecir["ma_12"])
        out["filas_entreno"] = len(entreno)
        resultados.append(out)
    return pd.concat(resultados, ignore_index=True)


def resumir(pred: pd.DataFrame) -> dict[str, float]:
    """Métricas globales de unas predicciones: MAE, RMSE, devianza y Capture@10 % medio mensual."""
    capturas = [
        capture_at_k(m["y"], m["yhat"], m["cell_id"]) for _, m in pred.groupby("target_month")
    ]
    return {
        "filas": len(pred),
        "MAE": mae(pred["y"], pred["yhat"]),
        "RMSE": rmse(pred["y"], pred["yhat"]),
        "devianza_poisson": devianza_poisson(pred["y"], pred["yhat"]),
        "capture@10%": float(np.nanmean(capturas)),
    }


def resumir_por_banda(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE y RMSE por banda de frecuencia histórica (Capture no aplica dentro de una banda)."""
    filas = []
    for banda, d in pred.groupby("banda", observed=False):
        filas.append(
            {
                "banda": banda,
                "filas": len(d),
                "MAE": mae(d["y"], d["yhat"]) if len(d) else np.nan,
                "RMSE": rmse(d["y"], d["yhat"]) if len(d) else np.nan,
            }
        )
    return pd.DataFrame(filas)


def resumir_por_mes(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE y Capture@10 % de cada mes evaluado."""
    filas = []
    for mes, d in pred.groupby("target_month"):
        filas.append(
            {
                "target_month": mes,
                "MAE": mae(d["y"], d["yhat"]),
                "RMSE": rmse(d["y"], d["yhat"]),
                "capture@10%": capture_at_k(d["y"], d["yhat"], d["cell_id"]),
            }
        )
    return pd.DataFrame(filas)


def baseline_como_modelo(columna: str) -> AjustarPredecir:
    """Un baseline de E0 visto como "modelo": no entrena y predice con la columna indicada.

    Sirve para pasar E0 por el mismo protocolo que E1–E5 y compararlos en igualdad.
    """

    def ajustar_predecir(entreno: pd.DataFrame, a_predecir: pd.DataFrame) -> np.ndarray:
        return a_predecir[columna].to_numpy(dtype=float)

    return ajustar_predecir

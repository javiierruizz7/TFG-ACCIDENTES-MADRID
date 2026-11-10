"""Métricas: MAE, RMSE, devianza de Poisson y Capture@K."""

import math

import numpy as np

EPS = 1e-8


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def devianza_poisson(y, yhat) -> float:
    """Devianza media de Poisson. Las predicciones se recortan a 1e-8 para no dividir por 0."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(yhat, float), EPS, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        termino = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2.0 * (termino - (y - mu))))


def capture_at_k(y, yhat, cell_ids, frac: float = 0.10) -> float:
    """Fracción de los accidentes reales que caen en el `frac` de celdas con mayor predicción.

    Se cogen k = ceil(frac * n) celdas. Empates en la predicción: gana el menor `cell_id`
    (así el resultado no depende del orden de las filas). Si ese mes no hay accidentes,
    devuelve NaN y hay que excluirlo al promediar.
    """
    y = np.asarray(y, float)
    total = y.sum()
    if total == 0:
        return math.nan
    k = math.ceil(frac * len(y))
    orden = np.lexsort((np.asarray(cell_ids), -np.asarray(yhat, float)))
    return float(y[orden[:k]].sum() / total)

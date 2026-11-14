"""E1: GLM Poisson con offset (regularización L2) sobre el historial local y el mes del año.

Modelo multiplicativo: mu = (ma_12 + c) · exp(b0 + b_mes + b·razones de historial)
es decir  log(mu) = log(ma_12 + c) + b0 + indicadoras de mes (+ log-razones de historial).

El offset log(ma_12 + c) hace que la predicción sea proporcional al nivel histórico de la celda.
Una primera versión con log1p(historial) como variables daba un modelo no proporcional (una celda
sin historial recibía ~0,23 accidentes) y quedaba por debajo de la media móvil: se descartó.

sklearn no admite offset: se usa la equivalencia de Poisson, ajustar y/e con peso e donde
e = ma_12 + c, y multiplicar la predicción por e.

Un único modelo global para todas las celdas; se pasa por `protocolo.evaluar_rolling`.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor

from protocolo import AjustarPredecir

C_OFFSET = 0.01  # constante dentro del log; elegida en validación 2024 (probadas 0,001–0,25)
# La COVID (2020) distorsiona esas filas y también las de 2021, cuya ventana de 12 meses de
# historial incluye meses de 2020. Se entrena solo con filas de 2022 en adelante. Decisión
# tomada mirando la validación 2024 (permitido); comparada con entrenar con todo (ver STATUS).
ENTRENO_DESDE = pd.Timestamp("2022-01-01")
RAZONES = ["lag_1", "lag_12", "ma_3", "ma_6"]


def exposicion(datos: pd.DataFrame, c: float = C_OFFSET) -> np.ndarray:
    """Nivel histórico de la fila: ma_12 + c (el offset es su logaritmo)."""
    return (datos["ma_12"].astype(float) + c).to_numpy()


def matriz_diseno(
    datos: pd.DataFrame, con_historial: bool = False, c: float = C_OFFSET
) -> pd.DataFrame:
    """11 indicadoras de mes (enero es la referencia) y, opcionalmente, las log-razones
    log((x + c) / (ma_12 + c)) de lag_1, lag_12, ma_3 y ma_6 respecto a la media de 12 meses."""
    meses = datos["target_month"].dt.month
    X = pd.DataFrame(index=datos.index)
    if con_historial:
        for v in RAZONES:
            X[f"razon_{v}"] = np.log((datos[v].astype(float) + c) / (datos["ma_12"] + c))
    for m in range(2, 13):
        X[f"mes_{m:02d}"] = (meses == m).astype(float)
    return X


def glm_poisson(
    con_historial: bool = False,
    alpha: float = 1e-3,
    c: float = C_OFFSET,
    entreno_desde: pd.Timestamp | None = ENTRENO_DESDE,
) -> tuple[AjustarPredecir, list[str]]:
    """Devuelve `(ajustar_predecir, columnas_necesarias)` listo para `evaluar_rolling`.

    `alpha` es la penalización L2; `con_historial=False` es el E1 principal (offset + mes).
    `entreno_desde` descarta del entrenamiento las filas anteriores (COVID); None = usar todo.
    """
    columnas = ["ma_12", *RAZONES] if con_historial else ["ma_12"]

    def ajustar_predecir(entreno: pd.DataFrame, a_predecir: pd.DataFrame) -> np.ndarray:
        if entreno_desde is not None:
            entreno = entreno[entreno["target_month"] >= entreno_desde]
        e = exposicion(entreno, c)
        modelo = PoissonRegressor(alpha=alpha, max_iter=3000, tol=1e-8)
        modelo.fit(matriz_diseno(entreno, con_historial, c), entreno["y"] / e, sample_weight=e)
        return modelo.predict(matriz_diseno(a_predecir, con_historial, c)) * exposicion(
            a_predecir, c
        )

    return ajustar_predecir, columnas


def contraste_sobredispersion(y, mu) -> dict[str, float]:
    """Contraste de sobredispersión de Cameron y Trivedi y razón de Pearson.

    Con Poisson, Var(y) = mu. Se estima alfa en Var(y) = mu + alfa·mu² con la regresión auxiliar
    ((y - mu)² - y) / mu = alfa · mu + error (sin constante). Alfa > 0 con p pequeño indica
    sobredispersión. La razón de Pearson (chi² / n) debería rondar 1.
    """
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-8, None)
    aux = ((y - mu) ** 2 - y) / mu
    ajuste = sm.OLS(aux, mu).fit()
    return {
        "alfa": float(ajuste.params[0]),
        "t": float(ajuste.tvalues[0]),
        "p": float(ajuste.pvalues[0]),
        "pearson_chi2_por_n": float(np.sum((y - mu) ** 2 / mu) / len(y)),
    }

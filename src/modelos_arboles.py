"""E2: modelos de árboles (LightGBM Poisson y Random Forest) con historial local y calendario.

Variables de cada fila (celda, mes m), todas conocidas antes de m: `lag_1`, `lag_12`, `ma_3`,
`ma_6`, `ma_12` y el mes del año. Un único modelo global para todas las celdas. Se pasan por
`protocolo.evaluar_rolling`, que reentrena cada mes solo con etiquetas ya conocidas.

LightGBM admite dos formas:
- `con_offset=False`: aprende `y` directamente a partir de las variables.
- `con_offset=True`: parte de `init_score = log(ma_12 + c)` (mismo offset que E1) y los árboles
  solo corrigen esa base. Así la predicción nace proporcional al historial de la celda.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from protocolo import AjustarPredecir

C_OFFSET = 0.01  # misma constante que E1
VARIABLES = ["lag_1", "lag_12", "ma_3", "ma_6", "ma_12"]
# Mismo criterio que E1: la COVID distorsiona 2020 y las filas de 2021 (su ventana de 12 meses
# incluye 2020). Se entrena con filas de 2022 en adelante; None = usar todo.
ENTRENO_DESDE = pd.Timestamp("2022-01-01")

PARAMS_LGBM = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
}


def matriz(datos: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Variables de historial, mes del año (1–12) y, opcionalmente, columnas `extra` (E3)."""
    X = datos[[*VARIABLES, *extra]].astype(float).copy()
    X["mes"] = datos["target_month"].dt.month.astype(float)
    return X


def _filtrar(entreno: pd.DataFrame, entreno_desde: pd.Timestamp | None) -> pd.DataFrame:
    if entreno_desde is None:
        return entreno
    return entreno[entreno["target_month"] >= entreno_desde]


def lightgbm_poisson(
    params: dict | None = None,
    con_offset: bool = False,
    c: float = C_OFFSET,
    entreno_desde: pd.Timestamp | None = ENTRENO_DESDE,
    n_jobs: int = 4,
    extra: tuple[str, ...] = (),
) -> tuple[AjustarPredecir, list[str]]:
    """LightGBM con `objective="poisson"`, listo para `evaluar_rolling`.

    `extra`: columnas adicionales del panel usadas como variables (E3, territoriales)."""
    p = {**PARAMS_LGBM, **(params or {})}

    def ajustar_predecir(entreno: pd.DataFrame, a_predecir: pd.DataFrame) -> np.ndarray:
        entreno = _filtrar(entreno, entreno_desde)
        modelo = lgb.LGBMRegressor(
            objective="poisson",
            random_state=0,
            n_jobs=n_jobs,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            **p,
        )
        if con_offset:
            base = np.log(entreno["ma_12"].to_numpy(float) + c)
            modelo.fit(matriz(entreno, extra), entreno["y"], init_score=base)
            raw = modelo.predict(matriz(a_predecir, extra), raw_score=True)
            return np.exp(raw + np.log(a_predecir["ma_12"].to_numpy(float) + c))
        modelo.fit(matriz(entreno, extra), entreno["y"])
        return modelo.predict(matriz(a_predecir, extra))

    return ajustar_predecir, [*VARIABLES, *extra]


def random_forest(
    n_estimators: int = 200,
    min_samples_leaf: int = 20,
    max_features: float = 1.0,
    entreno_desde: pd.Timestamp | None = ENTRENO_DESDE,
    n_jobs: int = 4,
) -> tuple[AjustarPredecir, list[str]]:
    """Random Forest (criterio Poisson), referencia de E2."""

    def ajustar_predecir(entreno: pd.DataFrame, a_predecir: pd.DataFrame) -> np.ndarray:
        entreno = _filtrar(entreno, entreno_desde)
        modelo = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion="poisson",
            random_state=0,
            n_jobs=n_jobs,
        )
        modelo.fit(matriz(entreno), entreno["y"])
        return modelo.predict(matriz(a_predecir))

    return ajustar_predecir, VARIABLES

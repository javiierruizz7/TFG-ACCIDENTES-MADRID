import numpy as np
import pandas as pd
import pytest

from features import anadir_rezagos
from modelos_arboles import lightgbm_poisson, matriz, random_forest
from protocolo import evaluar_rolling

MESES = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
EVAL = pd.date_range("2023-01-01", "2023-04-01", freq="MS")
RAPIDO = {"n_estimators": 30, "num_leaves": 4, "min_child_samples": 5}


def _panel(celdas=(1, 2, 3, 4)):
    rng = np.random.default_rng(0)
    filas = [
        (c, m, rng.poisson((1.0 + 2 * c) * (0.5 if m.month == 8 else 1.0)))
        for c in celdas
        for m in MESES
    ]
    return pd.DataFrame(filas, columns=["cell_id", "target_month", "y"])


def _datos(p):
    return anadir_rezagos(p, lags=(1, 12), ventanas=(3, 6, 12))


def _modelos():
    return {
        "lgbm": lightgbm_poisson(RAPIDO, n_jobs=1),
        "lgbm_offset": lightgbm_poisson(RAPIDO, con_offset=True, n_jobs=1),
        "rf": random_forest(n_estimators=10, min_samples_leaf=3, n_jobs=1),
    }


def test_matriz_tiene_historial_y_mes():
    X = matriz(_datos(_panel()).dropna())
    assert list(X.columns) == ["lag_1", "lag_12", "ma_3", "ma_6", "ma_12", "mes"]
    assert X["mes"].between(1, 12).all()


@pytest.mark.parametrize("nombre", ["lgbm", "lgbm_offset", "rf"])
def test_predicciones_finitas_y_no_negativas(nombre):
    modelo, cols = _modelos()[nombre]
    pred = evaluar_rolling(_datos(_panel()), EVAL, modelo, cols)
    assert np.isfinite(pred["yhat"]).all()
    assert (pred["yhat"] >= 0).all()


@pytest.mark.parametrize("nombre", ["lgbm", "lgbm_offset", "rf"])
def test_alterar_etiquetas_futuras_no_cambia_la_prediccion(nombre):
    p = _panel()
    modelo, cols = _modelos()[nombre]
    base = evaluar_rolling(_datos(p), EVAL, modelo, cols)
    mes = EVAL[2]
    alt = p.copy()
    alt.loc[alt["target_month"] >= mes, "y"] = 99
    otro = evaluar_rolling(_datos(alt), EVAL, modelo, cols)
    hasta = base["target_month"] <= mes
    np.testing.assert_allclose(base.loc[hasta, "yhat"], otro.loc[hasta, "yhat"])


def test_lightgbm_es_determinista():
    p = _datos(_panel())
    modelo, cols = lightgbm_poisson(RAPIDO, n_jobs=1)
    a = evaluar_rolling(p, EVAL, modelo, cols)["yhat"]
    b = evaluar_rolling(p, EVAL, modelo, cols)["yhat"]
    np.testing.assert_array_equal(a, b)


def test_con_offset_una_celda_sin_historial_recibe_casi_cero():
    p = _panel()
    p.loc[p["cell_id"] == 1, "y"] = 0
    modelo, cols = lightgbm_poisson(RAPIDO, con_offset=True, n_jobs=1)
    pred = evaluar_rolling(_datos(p), EVAL, modelo, cols)
    assert pred.loc[pred["cell_id"] == 1, "yhat"].max() < 0.1

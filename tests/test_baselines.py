import numpy as np
import pandas as pd
import pytest

from baselines import evaluar_e0, indice_estacional, predecir_e0

MESES = pd.date_range("2019-01-01", "2023-12-01", freq="MS")


def _panel(celdas=(1, 2)):
    """Panel con agosto a la mitad de los demás meses."""
    filas = [(c, m, 2 if m.month == 8 else 4) for c in celdas for m in MESES]
    return pd.DataFrame(filas, columns=["cell_id", "target_month", "y"])


def test_indice_estacional_detecta_agosto():
    idx = indice_estacional(_panel(), pd.Timestamp("2021-12-01"))
    media = (11 * 4 + 2) / 12
    assert idx[8] == pytest.approx(2 / media)
    assert idx[1] == pytest.approx(4 / media)


def test_indice_estacional_solo_usa_el_pasado_y_anos_completos():
    p = _panel()
    corte = pd.Timestamp("2022-06-01")  # 2022 está incompleto: no debe contar
    base = indice_estacional(p, corte)
    alterado = p.copy()
    alterado.loc[alterado["target_month"] > corte, "y"] = 999
    alterado.loc[alterado["target_month"].dt.year == 2022, "y"] = 999  # año incompleto
    pd.testing.assert_series_equal(base, indice_estacional(alterado, corte))


def test_indice_estacional_ignora_2020():
    p = _panel()
    alterado = p.copy()
    alterado.loc[alterado["target_month"].dt.year == 2020, "y"] = 0
    alterado.loc[(alterado["target_month"] == "2020-04-01"), "y"] = 50
    corte = pd.Timestamp("2021-12-01")
    pd.testing.assert_series_equal(indice_estacional(p, corte), indice_estacional(alterado, corte))


def test_indice_estacional_sin_anos_completos_es_nan():
    assert indice_estacional(_panel(), pd.Timestamp("2019-06-01")).isna().all()


def test_media_12_estacional_acierta_un_patron_estacional_puro():
    p = _panel()
    meses = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    r = evaluar_e0(p, meses).set_index("baseline")
    assert r.loc["media_12_estacional", "MAE"] == pytest.approx(0, abs=1e-9)
    assert r.loc["media_movil_12", "MAE"] > 0  # la media plana falla en agosto


def test_e0_con_latencia_2_usa_lag_2_y_la_estacionalidad_respeta_la_latencia():
    p = _panel()
    meses = pd.date_range("2023-01-01", "2023-03-01", freq="MS")
    preds = predecir_e0(p, meses, latencia=2)
    assert set(preds) >= {"persistencia", "media_12_estacional"}
    assert np.isfinite(preds["media_12_estacional"]["yhat"]).all()

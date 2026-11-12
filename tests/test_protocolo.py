import numpy as np
import pandas as pd
import pytest

from baselines import evaluar_e0
from features import anadir_rezagos
from protocolo import banda_frecuencia, baseline_como_modelo, evaluar_rolling, resumir

MESES = pd.date_range("2021-01-01", "2023-12-01", freq="MS")
EVAL = pd.date_range("2023-01-01", "2023-06-01", freq="MS")
COLS = ["lag_1", "lag_12", "ma_3", "ma_12"]


def _panel(celdas=(1, 2, 3)):
    filas = [(c, m, (i * 5 + c * 3) % 7) for c in celdas for i, m in enumerate(MESES)]
    return pd.DataFrame(filas, columns=["cell_id", "target_month", "y"])


def _datos(panel, latencia=1):
    lag = max(1, latencia)
    return anadir_rezagos(panel, lags=(lag, 12), ventanas=(3, 12), latencia=latencia)


def _cols(latencia):
    return [f"lag_{max(1, latencia)}", "lag_12", "ma_3", "ma_12"]


def _media_del_entreno(entreno, a_predecir):
    """Modelo de juguete que sí aprende algo de las etiquetas de entrenamiento."""
    return np.full(len(a_predecir), entreno["y"].mean())


@pytest.mark.parametrize("latencia", [1, 2])
def test_el_entrenamiento_solo_ve_meses_conocidos(latencia):
    vistos = []

    def espia(entreno, a_predecir):
        vistos.append((a_predecir["target_month"].unique(), entreno["target_month"].max()))
        return np.zeros(len(a_predecir))

    evaluar_rolling(_datos(_panel(), latencia), EVAL, espia, _cols(latencia), latencia)
    assert len(vistos) == len(EVAL)
    for (meses_pred, max_entreno), mes in zip(vistos, EVAL, strict=True):
        assert list(meses_pred) == [mes]  # se predice un único mes cada vez
        assert max_entreno == mes - pd.DateOffset(months=latencia)  # y nada más reciente


def test_se_reentrena_cada_mes_con_un_mes_mas():
    pred = evaluar_rolling(_datos(_panel()), EVAL, _media_del_entreno, COLS)
    por_mes = pred.groupby("target_month")["filas_entreno"].first()
    assert (por_mes.diff().dropna() == 3).all()  # 3 celdas = un mes más cada vez


def test_cambiar_etiquetas_futuras_no_cambia_la_prediccion():
    """Anti-fuga a nivel de protocolo: alterar y en meses >= m no cambia la predicción de m."""
    p = _panel()
    base = evaluar_rolling(_datos(p), EVAL, _media_del_entreno, COLS)
    mes = EVAL[3]
    alterado = p.copy()
    alterado.loc[alterado["target_month"] >= mes, "y"] = 999
    otro = evaluar_rolling(_datos(alterado), EVAL, _media_del_entreno, COLS)
    hasta = base["target_month"] <= mes
    np.testing.assert_array_equal(base.loc[hasta, "yhat"], otro.loc[hasta, "yhat"])


def test_e0_por_el_protocolo_da_lo_mismo_que_evaluar_e0():
    p = _panel()
    directo = evaluar_e0(p, EVAL).set_index("baseline")
    datos = anadir_rezagos(p, lags=(1, 12), ventanas=(3, 6, 12))
    pred = evaluar_rolling(datos, EVAL, baseline_como_modelo("ma_12"), ["ma_12"])
    r = resumir(pred)
    assert r["MAE"] == pytest.approx(directo.loc["media_movil_12", "MAE"])
    assert r["devianza_poisson"] == pytest.approx(directo.loc["media_movil_12", "devianza_poisson"])


def test_bandas():
    ma = pd.Series([0.0, 0.1, 2.99, 3.0, 7.5, np.nan])
    b = banda_frecuencia(ma)
    assert list(b.astype(object)[:5]) == ["0", "1-2", "1-2", ">=3", ">=3"]
    assert pd.isna(b.iloc[5])


def test_mes_sin_filas_falla():
    with pytest.raises(ValueError):
        evaluar_rolling(
            _datos(_panel()), pd.DatetimeIndex(["2030-01-01"]), _media_del_entreno, COLS
        )

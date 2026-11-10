import numpy as np
import pandas as pd
import pytest

from features import anadir_rezagos

MESES = pd.date_range("2022-01-01", "2023-12-01", freq="MS")


def _panel(celdas=(1, 2)):
    filas = [(c, m, (i * 7 + c * 3) % 11) for c in celdas for i, m in enumerate(MESES)]
    return pd.DataFrame(filas, columns=["cell_id", "target_month", "y"])


def _y(panel, celda, mes):
    return panel.loc[(panel.cell_id == celda) & (panel.target_month == mes), "y"].item()


def test_rezagos_solo_miran_al_pasado():
    p = _panel()
    f = anadir_rezagos(p, lags=(1, 12), ventanas=(3,), latencia=1)
    mes = MESES[15]
    for celda in (1, 2):
        fila = f[(f.cell_id == celda) & (f.target_month == mes)].iloc[0]
        assert fila["lag_1"] == _y(p, celda, MESES[14])
        assert fila["lag_12"] == _y(p, celda, MESES[3])
        assert fila["ma_3"] == np.mean([_y(p, celda, MESES[i]) for i in (14, 13, 12)])


def test_latencia_2_no_usa_el_mes_anterior():
    p = _panel()
    f = anadir_rezagos(p, lags=(2,), ventanas=(3,), latencia=2)
    mes = MESES[15]
    fila = f[(f.cell_id == 1) & (f.target_month == mes)].iloc[0]
    assert fila["lag_2"] == _y(p, 1, MESES[13])
    assert fila["ma_3"] == np.mean([_y(p, 1, MESES[i]) for i in (13, 12, 11)])


def test_cambiar_el_futuro_no_cambia_las_variables():
    """Prueba clave anti-fuga: si se alteran los meses >= m, las variables de m no cambian."""
    p = _panel()
    mes = MESES[15]
    p2 = p.copy()
    p2.loc[p2.target_month >= mes, "y"] = 999
    cols = ["lag_1", "lag_12", "ma_3", "ma_6", "ma_12"]
    a = anadir_rezagos(p)
    b = anadir_rezagos(p2)
    pd.testing.assert_frame_equal(
        a[a.target_month == mes][cols].reset_index(drop=True),
        b[b.target_month == mes][cols].reset_index(drop=True),
    )


def test_primeros_meses_son_nan_y_no_se_inventa_historia():
    f = anadir_rezagos(_panel(), lags=(12,), ventanas=(), latencia=1)
    assert f[f.target_month < MESES[12]]["lag_12"].isna().all()
    assert f[f.target_month >= MESES[12]]["lag_12"].notna().all()


def test_no_mezcla_celdas():
    p = _panel()
    f = anadir_rezagos(p, lags=(1,), ventanas=(), latencia=1)
    primera = f[f.target_month == MESES[0]]
    assert primera["lag_1"].isna().all()  # el primer mes de cada celda no hereda de otra


def test_lag_menor_que_la_latencia_falla():
    with pytest.raises(ValueError):
        anadir_rezagos(_panel(), lags=(1,), latencia=2)


def test_panel_con_mes_que_falta_falla():
    p = _panel()
    p = p[~((p.cell_id == 1) & (p.target_month == MESES[5]))]
    with pytest.raises(ValueError):
        anadir_rezagos(p)


def test_e0_con_latencia_2_usa_lag_2_para_persistencia_y_lag_12_para_estacionalidad():
    from baselines import evaluar_e0

    p = _panel()
    r = evaluar_e0(p, MESES[14:], latencia=2).set_index("baseline")
    f = anadir_rezagos(p, lags=(2, 12), ventanas=(3, 6, 12), latencia=2)
    d = f[f.target_month.isin(MESES[14:])]
    assert np.isclose(r.loc["persistencia", "MAE"], (d["y"] - d["lag_2"]).abs().mean())
    assert np.isclose(r.loc["estacionalidad", "MAE"], (d["y"] - d["lag_12"]).abs().mean())

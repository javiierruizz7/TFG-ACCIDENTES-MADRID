import numpy as np
import pandas as pd
import pytest

from features import anadir_rezagos
from modelos_arboles import lightgbm_poisson
from protocolo import evaluar_rolling
from territorio import GRUPOS, con_territorio, tabla_territorial

MESES = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
EVAL = pd.date_range("2023-01-01", "2023-03-01", freq="MS")


@pytest.mark.parametrize("res", ["1km", "2km", "distritos"])
def test_tabla_territorial_una_fila_por_zona_sin_nulos(res):
    t = tabla_territorial(res)
    assert t["cell_id"].is_unique
    assert not t.isna().any().any()
    assert t["distrito"].between(1, 21).all()
    assert (t["area_km2"] > 0).all() and (t["frac_dentro"] <= 1 + 1e-9).all()


def test_con_territorio_no_toca_el_conteo_ni_las_filas():
    p = pd.DataFrame(
        {"cell_id": [0, 1, 0, 1], "target_month": MESES[:2].repeat(2), "y": [1, 2, 3, 4]}
    )
    out = con_territorio(p, "1km")
    assert len(out) == len(p)
    assert out["y"].tolist() == [1, 2, 3, 4]


def test_e3_no_usa_el_futuro_y_usa_las_columnas_extra():
    rng = np.random.default_rng(0)
    filas = [(c, m, rng.poisson(1 + 2 * c)) for c in (1, 2, 3, 4) for m in MESES]
    p = pd.DataFrame(filas, columns=["cell_id", "target_month", "y"])
    p["coord_x"] = p["cell_id"] * 1.5  # variable estática de prueba
    extra = GRUPOS["posicion"][:1]
    modelo, cols = lightgbm_poisson(
        {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 5},
        con_offset=True,
        n_jobs=1,
        extra=extra,
    )
    assert cols[-1] == "coord_x"
    base = evaluar_rolling(anadir_rezagos(p, lags=(1, 12), ventanas=(3, 6, 12)), EVAL, modelo, cols)
    alt = p.copy()
    alt.loc[alt["target_month"] >= EVAL[1], "y"] = 99
    otro = evaluar_rolling(
        anadir_rezagos(alt, lags=(1, 12), ventanas=(3, 6, 12)), EVAL, modelo, cols
    )
    hasta = base["target_month"] <= EVAL[1]
    np.testing.assert_allclose(base.loc[hasta, "yhat"], otro.loc[hasta, "yhat"])

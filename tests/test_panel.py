import pandas as pd
import pytest

from panel import construir_panel

MESES = pd.date_range("2023-01-01", "2023-06-01", freq="MS")


def _asignacion():
    return pd.DataFrame(
        {"cell": [1, 1, 2, 3, 3, 3], "mes": [MESES[0], MESES[0], MESES[2]] + [MESES[5]] * 3}
    )


def test_producto_cartesiano_con_ceros_y_conservacion():
    celdas = pd.Series([1, 2, 3, 4])
    p = construir_panel(_asignacion(), celdas, "cell", MESES)
    assert len(p) == 4 * 6
    assert p["y"].sum() == 6
    assert (p["y"] == 0).sum() == 24 - 3
    assert "geometry" not in p.columns


def test_accidente_fuera_del_panel_falla():
    with pytest.raises(ValueError):
        construir_panel(_asignacion(), pd.Series([1, 2]), "cell", MESES)

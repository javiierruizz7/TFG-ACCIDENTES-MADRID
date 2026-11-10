"""Rezagos y medias móviles sin fuga de información.

Para predecir el mes m solo se usan meses <= m - latencia. Con latencia=1 se supone que ya
conocemos el mes anterior; con 2, que los datos tardan un mes más en publicarse.
"""

import pandas as pd


def _comprobar_panel_completo(panel: pd.DataFrame) -> None:
    """El desplazamiento por filas solo vale si cada celda tiene todos los meses seguidos."""
    meses = panel["target_month"].drop_duplicates().sort_values()
    saltos = meses.diff().dropna().dt.days
    if not saltos.between(28, 31).all():
        raise ValueError("Faltan meses en el panel: los meses no son consecutivos")
    if (panel.groupby("cell_id").size() != len(meses)).any():
        raise ValueError("Hay celdas sin todos los meses del panel")


def anadir_rezagos(
    panel: pd.DataFrame,
    lags: tuple[int, ...] = (1, 12),
    ventanas: tuple[int, ...] = (3, 6, 12),
    latencia: int = 1,
) -> pd.DataFrame:
    """Añade `lag_k` (y de hace k meses) y `ma_w` (media de los w meses que acaban en m - latencia).

    Devuelve NaN donde no hay suficiente historia dentro del panel.
    """
    if latencia < 1:
        raise ValueError("latencia tiene que ser >= 1")
    if any(k < latencia for k in lags):
        raise ValueError("Ningún lag puede ser menor que la latencia")
    _comprobar_panel_completo(panel)

    out = panel.sort_values(["cell_id", "target_month"]).reset_index(drop=True)
    y = out.groupby("cell_id")["y"]
    for k in lags:
        out[f"lag_{k}"] = y.shift(k)
    for w in ventanas:
        out[f"ma_{w}"] = y.transform(
            lambda s, w=w: s.shift(latencia).rolling(w, min_periods=w).mean()
        )
    return out

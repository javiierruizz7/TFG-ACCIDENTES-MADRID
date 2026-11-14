import numpy as np
import pandas as pd
import pytest

from features import anadir_rezagos
from modelos_conteo import contraste_sobredispersion, exposicion, glm_poisson, matriz_diseno
from protocolo import evaluar_rolling

MESES = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
EVAL = pd.date_range("2023-01-01", "2023-06-01", freq="MS")


def _panel(celdas=(1, 2, 3, 4)):
    rng = np.random.default_rng(0)
    filas = []
    for c in celdas:
        for m in MESES:
            base = 1.0 + 2 * c
            filas.append((c, m, rng.poisson(base * (0.5 if m.month == 8 else 1.0))))
    return pd.DataFrame(filas, columns=["cell_id", "target_month", "y"])


def _datos(p):
    return anadir_rezagos(p, lags=(1, 12), ventanas=(3, 6, 12))


def test_matriz_de_diseno_tiene_11_indicadoras_de_mes():
    d = _datos(_panel()).dropna()
    X = matriz_diseno(d)
    assert list(X.columns) == [f"mes_{m:02d}" for m in range(2, 13)]
    assert (X.loc[d["target_month"].dt.month == 1] == 0).all().all()
    assert matriz_diseno(d, con_historial=True).shape[1] == 11 + 4


def test_exposicion_es_ma12_mas_constante():
    d = _datos(_panel()).dropna()
    np.testing.assert_allclose(exposicion(d, c=0.5), d["ma_12"].to_numpy() + 0.5)


def test_glm_es_proporcional_al_historial_y_aprende_el_efecto_agosto():
    p = _panel()
    modelo, cols = glm_poisson(alpha=1e-4)
    meses = pd.date_range("2023-07-01", "2023-09-01", freq="MS")
    pred = evaluar_rolling(_datos(p), meses, modelo, cols)
    assert (pred["yhat"] > 0).all()
    medias = pred.groupby(pred["target_month"].dt.month)["yhat"].mean()
    assert medias[8] < 0.8 * medias[7]  # agosto con ~la mitad de accidentes
    # celdas con más historial reciben más predicción
    julio = pred[pred["target_month"] == "2023-07-01"].set_index("cell_id")["yhat"]
    assert julio.is_monotonic_increasing


def test_celda_sin_historial_recibe_casi_cero():
    p = _panel()
    p.loc[p["cell_id"] == 1, "y"] = 0
    modelo, cols = glm_poisson()
    pred = evaluar_rolling(_datos(p), EVAL, modelo, cols)
    assert pred.loc[pred["cell_id"] == 1, "yhat"].max() < 0.05


def test_alterar_etiquetas_futuras_no_cambia_la_prediccion_del_glm():
    p = _panel()
    for con in (False, True):
        modelo, cols = glm_poisson(con_historial=con)
        base = evaluar_rolling(_datos(p), EVAL, modelo, cols)
        mes = EVAL[3]
        alt = p.copy()
        alt.loc[alt["target_month"] >= mes, "y"] = 99
        otro = evaluar_rolling(_datos(alt), EVAL, modelo, cols)
        hasta = base["target_month"] <= mes
        np.testing.assert_allclose(base.loc[hasta, "yhat"], otro.loc[hasta, "yhat"])


def test_contraste_sobredispersion_detecta_datos_sobredispersos_y_no_los_poisson():
    rng = np.random.default_rng(1)
    mu = rng.uniform(1, 6, 20000)
    poisson = rng.poisson(mu)
    r = 1 / 0.5  # binomial negativa con alfa = 0,5: Var = mu + 0,5·mu²
    nb = rng.negative_binomial(r, r / (r + mu))
    assert contraste_sobredispersion(poisson, mu)["p"] > 0.01
    sob = contraste_sobredispersion(nb, mu)
    assert sob["p"] < 1e-6
    assert sob["alfa"] == pytest.approx(0.5, abs=0.1)


def test_entreno_desde_descarta_filas_anteriores():
    vistos = []
    modelo, cols = glm_poisson(entreno_desde=pd.Timestamp("2022-01-01"))
    p = _panel()
    d = _datos(p)

    def espia(entreno, a_predecir):
        vistos.append(entreno["target_month"].min())
        return modelo(entreno, a_predecir)

    evaluar_rolling(d, EVAL, espia, cols)
    assert all(v < pd.Timestamp("2022-01-01") for v in vistos)  # el protocolo sí pasa todo
    # y el modelo, por dentro, solo ajusta con 2022 en adelante: igual que pasárselo ya filtrado
    filtrado = d[d["target_month"] >= "2022-01-01"]
    a = modelo(d[d["target_month"] < "2023-01-01"], d[d["target_month"] == "2023-01-01"].dropna())
    b = modelo(filtrado[filtrado["target_month"] < "2023-01-01"].dropna(),
               d[d["target_month"] == "2023-01-01"].dropna())
    np.testing.assert_allclose(a, b)

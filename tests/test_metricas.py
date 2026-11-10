import math

import numpy as np

from metricas import capture_at_k, devianza_poisson, mae, rmse


def test_errores_basicos():
    y, p = np.array([0, 2, 4]), np.array([1, 2, 2])
    assert mae(y, p) == 1.0
    assert math.isclose(rmse(y, p), math.sqrt(5 / 3))


def test_devianza_con_prediccion_cero_es_finita_y_perfecta_es_cero():
    assert math.isfinite(devianza_poisson(np.array([0, 3]), np.array([0.0, 0.0])))
    perfecta = devianza_poisson(np.array([1, 2]), np.array([1.0, 2.0]))
    assert math.isclose(perfecta, 0.0, abs_tol=1e-12)


def test_capture_basico():
    y = np.array([5, 0, 1, 0, 0, 0, 0, 0, 4, 0])
    ids = np.arange(10)
    assert capture_at_k(y, np.arange(10.0), ids, 0.10) == 0.0
    assert capture_at_k(y, np.array([9, 0, 1, 0, 0, 0, 0, 0, 2, 0.0]), ids, 0.10) == 0.5


def test_capture_empates_por_cell_id_y_sin_depender_del_orden():
    y = np.array([3, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    yhat, ids = np.ones(10), np.arange(10)
    assert capture_at_k(y, yhat, ids, 0.10) == 0.75
    perm = np.random.default_rng(0).permutation(10)
    assert capture_at_k(y[perm], yhat[perm], ids[perm], 0.10) == 0.75


def test_capture_con_mes_sin_accidentes_es_nan():
    assert math.isnan(capture_at_k(np.zeros(10), np.arange(10.0), np.arange(10), 0.10))

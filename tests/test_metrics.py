import numpy as np

from model.metrics import k_error_pcm, peaking_factor, power_rel_error


def test_k_error_pcm():
    assert np.isclose(k_error_pcm(np.array(1.001), np.array(1.0)), 100.0)


def test_power_metrics():
    p = np.array([[1.0, 2.0], [1.0, 0.0]])
    mask = np.array([[True, True], [True, False]])
    assert np.isclose(peaking_factor(p, mask), 1.5)
    assert power_rel_error(p, p, mask)["max"] == 0.0

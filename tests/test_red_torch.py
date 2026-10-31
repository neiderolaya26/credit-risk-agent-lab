import numpy as np

from credit_lab.red_torch import RedTorch


def test_red_aprende_un_patron_simple():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=4000) > 0).astype(int)
    red = RedTorch(capas=(16,), max_epocas=20, paciencia=3).fit(X, y)
    p = red.predict_proba(X)
    assert p.shape == (4000, 2)
    assert np.allclose(p.sum(axis=1), 1)
    assert ((p[:, 1] > 0.5) == y).mean() > 0.8

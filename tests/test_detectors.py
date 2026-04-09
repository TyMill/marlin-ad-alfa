import numpy as np

from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.detection.classical.robust_stats import RobustZScoreDetector


def test_isolation_forest_shapes():
    rng = np.random.default_rng(1)
    X_ref = rng.normal(size=(128, 3))
    X_cur = rng.normal(size=(32, 3))
    det = IsolationForestDetector(random_state=1).fit(X_ref)
    res = det.score(X_cur)
    assert res.scores.shape == (32,)
    assert res.labels is not None
    assert res.labels.shape == (32,)


def test_robust_zscore_flags_outlier():
    rng = np.random.default_rng(2)
    X_ref = rng.normal(size=(100, 2))
    outlier = np.array([[10.0, 10.0]])
    X_cur = np.vstack([rng.normal(size=(5, 2)), outlier])
    det = RobustZScoreDetector(threshold=3.5).fit(X_ref)
    res = det.score(X_cur)
    assert res.labels is not None
    assert res.labels[-1] == 1

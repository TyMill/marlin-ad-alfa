import numpy as np

from marlin_ad.detection.temporal.changepoint import cusum_changepoint
from marlin_ad.monitoring.uncertainty.conformal import ConformalRegressor
from marlin_ad.monitoring.uncertainty.ensembles import ensemble_mean_std


def test_cusum_changepoint_detects_shift():
    series = np.concatenate([np.zeros(50), np.ones(50) * 5.0])
    result = cusum_changepoint(series, threshold=1.0)
    assert result.indices.size > 0


def test_conformal_intervals_shapes():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.1, 0.9, 2.2])
    model = ConformalRegressor(alpha=0.1).fit(y_true, y_pred)
    intervals = model.interval(y_pred)
    assert intervals.lower.shape == y_pred.shape
    assert intervals.upper.shape == y_pred.shape


def test_ensemble_mean_std():
    preds = np.array([[0.0, 1.0], [1.0, 2.0]])
    stats = ensemble_mean_std(preds)
    assert np.allclose(stats.mean, [0.5, 1.5])

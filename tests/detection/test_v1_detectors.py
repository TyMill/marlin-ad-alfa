from __future__ import annotations

import numpy as np
import pytest

from marlin_ad.detection import labels_from_threshold, score_threshold, threshold_scores
from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.detection.classical.robust_stats import RobustZScoreDetector
from marlin_ad.types.errors import DataError


def test_score_threshold_and_labels_orientation() -> None:
    scores = np.array([0.1, 0.2, 0.3, 5.0], dtype=float)

    threshold = score_threshold(scores, quantile=0.75)
    labels_q = threshold_scores(scores, quantile=0.75)
    labels_fixed = labels_from_threshold(scores, threshold)

    assert threshold == pytest.approx(1.475)
    np.testing.assert_array_equal(labels_q, np.array([0, 0, 0, 1]))
    np.testing.assert_array_equal(labels_fixed, labels_q)


def test_threshold_utils_validate_inputs() -> None:
    with pytest.raises(DataError):
        score_threshold(np.array([]), quantile=0.95)

    with pytest.raises(DataError):
        threshold_scores(np.array([0.1, np.nan]), quantile=0.95)

    with pytest.raises(DataError):
        score_threshold(np.array([0.1, 0.2]), quantile=1.0)

    with pytest.raises(DataError):
        labels_from_threshold(np.array([0.1, 0.2]), threshold=np.inf)


def test_isolation_forest_shapes_labels_and_metadata() -> None:
    rng = np.random.default_rng(123)
    x_ref = rng.normal(size=(256, 5))
    x_cur = rng.normal(size=(32, 5))

    detector = IsolationForestDetector(random_state=42, quantile=0.9).fit(x_ref)
    result = detector.score(x_cur)

    assert result.scores.shape == (32,)
    assert result.labels is not None
    assert result.labels.shape == (32,)
    assert result.metadata["method"] == "isolation_forest"
    assert result.metadata["score_orientation"] == "higher_is_more_anomalous"

    threshold_meta = result.metadata["thresholding"]
    assert isinstance(threshold_meta, dict)
    assert threshold_meta["kind"] == "quantile"
    assert threshold_meta["quantile"] == pytest.approx(0.9)


def test_isolation_forest_deterministic_given_same_seed() -> None:
    rng = np.random.default_rng(77)
    x_ref = rng.normal(size=(200, 3))
    x_cur = rng.normal(size=(40, 3))

    detector_a = IsolationForestDetector(random_state=19, quantile=0.95).fit(x_ref)
    detector_b = IsolationForestDetector(random_state=19, quantile=0.95).fit(x_ref)

    result_a = detector_a.score(x_cur)
    result_b = detector_b.score(x_cur)

    np.testing.assert_allclose(result_a.scores, result_b.scores)
    np.testing.assert_array_equal(result_a.labels, result_b.labels)


def test_isolation_forest_feature_mismatch_raises() -> None:
    rng = np.random.default_rng(99)
    x_ref = rng.normal(size=(100, 4))
    x_bad = rng.normal(size=(4, 3))

    detector = IsolationForestDetector().fit(x_ref)
    with pytest.raises(DataError, match="Expected 4 features"):
        detector.score(x_bad)


def test_robust_zscore_flags_outlier_and_feature_scores_shape() -> None:
    rng = np.random.default_rng(3)
    x_ref = rng.normal(size=(150, 2))
    outlier = np.array([[12.0, -10.0]])
    x_cur = np.vstack([rng.normal(size=(10, 2)), outlier])

    detector = RobustZScoreDetector(threshold=3.5).fit(x_ref)
    result = detector.score(x_cur)

    assert result.labels is not None
    assert result.labels[-1] == 1
    assert result.scores.shape == (11,)

    assert result.explanations is not None
    feature_scores = result.explanations["feature_scores"]
    assert isinstance(feature_scores, np.ndarray)
    assert feature_scores.shape == (11, 2)


def test_robust_zscore_zero_variance_reference_is_stable() -> None:
    x_ref = np.ones((32, 2), dtype=float)
    x_cur = np.array([[1.0, 1.0], [1.0, 50.0]], dtype=float)

    detector = RobustZScoreDetector(threshold=3.5, epsilon=1e-9).fit(x_ref)
    result = detector.score(x_cur)

    assert np.isfinite(result.scores).all()
    assert result.labels is not None
    np.testing.assert_array_equal(result.labels, np.array([0, 1]))


def test_robust_zscore_feature_mismatch_raises() -> None:
    rng = np.random.default_rng(17)
    x_ref = rng.normal(size=(30, 2))
    x_bad = rng.normal(size=(5, 4))

    detector = RobustZScoreDetector().fit(x_ref)
    with pytest.raises(DataError, match="Expected 2 features"):
        detector.score(x_bad)


def test_detector_param_validation() -> None:
    with pytest.raises(DataError):
        IsolationForestDetector(n_estimators=0)

    with pytest.raises(DataError):
        IsolationForestDetector(contamination=0.8)

    with pytest.raises(DataError):
        RobustZScoreDetector(threshold=0.0)

    with pytest.raises(DataError):
        RobustZScoreDetector(epsilon=0.0)

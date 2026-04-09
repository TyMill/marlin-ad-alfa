import numpy as np
import pandas as pd
from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.monitoring.drift.data_drift import KSTestDriftMonitor

def test_end_to_end_smoke():
    rng = np.random.default_rng(0)
    X_ref = rng.normal(size=(200, 2))
    X_cur = rng.normal(loc=0.2, size=(200, 2))

    det = IsolationForestDetector().fit(X_ref)
    res = det.score(X_cur)
    assert res.scores is not None

    ref = pd.DataFrame(X_ref, columns=["a", "b"])
    cur = pd.DataFrame(X_cur, columns=["a", "b"])
    mon = KSTestDriftMonitor().fit(ref)
    out = mon.evaluate(cur)
    assert isinstance(out.metrics, dict)

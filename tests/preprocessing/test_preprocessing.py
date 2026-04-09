from __future__ import annotations

import pandas as pd
import pytest

from marlin_ad.preprocessing import add_rate_of_change, basic_clean, ensure_datetime, rolling_mean


def test_basic_clean_trims_text_drops_duplicate_and_missing_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", None],
            "vessel_id": [" V1 ", " V1 ", " V2 "],
            "rpm": [90, 90, 105],
        }
    )

    cleaned = basic_clean(raw)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "vessel_id"] == "V1"


def test_ensure_datetime_sorts_by_vessel_and_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T01:00:00Z", "2025-01-01T00:00:00Z"],
            "vessel_id": ["V1", "V1"],
        }
    )

    normalized = ensure_datetime(raw)
    assert normalized["timestamp"].is_monotonic_increasing


def test_add_rate_of_change_and_rolling_mean_are_composable() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:10:00Z",
                    "2025-01-01T00:20:00Z",
                ],
                utc=True,
            ),
            "vessel_id": ["V1", "V1", "V1"],
            "sog": [10.0, 11.0, 14.0],
        }
    )

    with_roc = add_rate_of_change(frame, "sog")
    with_roll = rolling_mean(with_roc, "sog", window=2)

    assert pd.isna(with_roc.loc[0, "sog_roc"])
    assert with_roc.loc[1:, "sog_roc"].tolist() == [1.0, 3.0]
    assert with_roll["sog_rollmean_2"].tolist() == pytest.approx([10.0, 10.5, 12.5])

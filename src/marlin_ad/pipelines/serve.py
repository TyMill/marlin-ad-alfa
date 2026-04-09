from __future__ import annotations

from typing import Any


def create_app(detector: Any) -> Any:
    """Create a minimal FastAPI app for serving detector scores.

    Requires optional dependency: `pip install marlin-ad[serve]`.
    """
    try:
        from fastapi import FastAPI  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("FastAPI is required for serving. Install marlin-ad[serve].") from exc

    app = FastAPI(title="MARLIN-AD Detector Service")

    @app.post("/score")  # type: ignore[untyped-decorator]
    def score(payload: dict[str, list[float]]) -> dict[str, Any]:
        scores = detector.score(payload["features"])
        return {"scores": scores.scores.tolist(), "labels": scores.labels.tolist() if scores.labels is not None else None}

    return app

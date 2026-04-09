from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MaritimeContext:
    vessel_id: Optional[str] = None
    timestamp_col: str = "timestamp"
    latitude_col: str = "lat"
    longitude_col: str = "lon"

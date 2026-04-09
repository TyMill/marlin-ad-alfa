from __future__ import annotations

from marlin_ad.alerting.formatter import Alert


def send(alert: Alert, url: str) -> None:
    """Send alert to a webhook endpoint.

    TODO: Implement when webhook delivery is required. This stub documents the expected interface:
    - `alert` contains the formatted payload.
    - `url` is the webhook endpoint URL.
    """
    raise NotImplementedError("Webhook delivery is not implemented yet.")

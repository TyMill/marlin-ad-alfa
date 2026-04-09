from __future__ import annotations

from marlin_ad.cli.main import main


def test_cli_demo_runs(capsys) -> None:
    main(["demo", "--rows", "120", "--seed", "7"])
    captured = capsys.readouterr()

    assert "Mode: synthetic demo" in captured.out
    assert "Anomaly rate:" in captured.out
    assert "Drift alerts:" in captured.out


def test_cli_datasets_list(capsys) -> None:
    main(["datasets", "list"])
    captured = capsys.readouterr()

    assert "ais:" in captured.out
    assert "engine_sensors:" in captured.out

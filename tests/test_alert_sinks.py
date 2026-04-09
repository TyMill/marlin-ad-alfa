from __future__ import annotations

from marlin_ad.alerting.formatter import Alert
from marlin_ad.alerting.sinks.file import FileAlertSink
from marlin_ad.alerting.sinks.stdout import StdoutAlertSink


def test_stdout_sink_writes_formatted_alert(capsys) -> None:
    sink = StdoutAlertSink()
    sink.send(Alert(name="demo", payload={"x": 1}, timestamp="2026-01-01T00:00:00+00:00"))

    captured = capsys.readouterr()
    assert "name=demo" in captured.out


def test_file_sink_writes_line(tmp_path) -> None:
    output = tmp_path / "alerts" / "alerts.log"
    sink = FileAlertSink(path=str(output))
    sink.send(Alert(name="demo", payload={"x": 1}, timestamp="2026-01-01T00:00:00+00:00"))

    content = output.read_text(encoding="utf-8")
    assert "name=demo" in content

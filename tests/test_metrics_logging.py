import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cre_advance.metrics import get_metrics, log_feedback, log_metric
from cre_advance.utils.logging import get_logger


def test_metrics_persistence(tmp_path, monkeypatch):
    from cre_advance import metrics as m

    monkeypatch.setattr(m, "_DB_PATH", tmp_path / "metrics.db")
    m._CONN = None

    log_metric("accuracy", 0.9, tags={"file": "a.xlsx"}, feedback={"corr": True})
    rows = get_metrics("accuracy")
    assert rows[0]["value"] == 0.9
    assert rows[0]["tags"] == {"file": "a.xlsx"}
    assert rows[0]["feedback"] == {"corr": True}

    log_feedback("accuracy", {"corr": False}, tags={"file": "a.xlsx"})
    rows = get_metrics("accuracy")
    assert len(rows) == 2


def test_logging_tags(caplog, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    logging.getLogger("test").handlers.clear()
    stream = io.StringIO()
    logger = get_logger("test")
    for h in logger.logger.handlers:
        h.stream = stream
    logger.info(
        "hello",
        extra={"context": {"file": "a.pdf", "page": 2, "invoice": "123"}},
    )
    out = stream.getvalue()
    assert "file=a.pdf" in out
    assert "page=2" in out
    assert "invoice=123" in out

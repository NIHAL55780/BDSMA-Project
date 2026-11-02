from __future__ import annotations

from lpdp.compliance.evaluate import evaluate_compliance, load_checklist, write_notice
from lpdp.config import load_config


def test_checklist_loads():
    items = load_checklist()
    ids = {i["id"] for i in items}
    assert {"consent", "minimisation", "security", "children"} <= ids


def test_notice_written(tmp_path):
    cfg = load_config("synthetic")
    path = write_notice(cfg, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "purpose" in text
    assert "fields" in text


def test_evaluate_writes_matrix(tmp_path, monkeypatch):
    df = evaluate_compliance("synthetic")
    assert not df.empty
    assert set(df["status"]) <= {"pass", "partial", "fail", "na"}
    assert "M0" in set(df["mechanism_id"])
    m0_sec = df[(df["mechanism_id"] == "M0") & (df["item_id"] == "security")].iloc[0]
    assert m0_sec["status"] == "fail"

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_mechanisms_and_perturb():
    r = client.get("/mechanisms")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()}
    assert {"M0", "M3", "M7"} <= ids
    p = client.post("/perturb", json={"mechanism": "M3", "param": 2.0, "n_points": 8})
    assert p.status_code == 200
    body = p.json()
    assert len(body["raw"]) == 8
    assert len(body["perturbed"]) == 8

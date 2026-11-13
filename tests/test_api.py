from fastapi.testclient import TestClient
from medreconlab.api import app

client = TestClient(app)


def test_health_and_methods():
    assert client.get("/health").status_code == 200
    assert len(client.get("/methods").json()["ct"]) >= 2


def test_reconstruction_and_report():
    response = client.post("/reconstruct", json={"modality": "mri", "size": 32, "noise": 0})
    assert response.status_code == 200
    run_id = response.json()["id"]
    assert client.get(f"/reports/{run_id}").status_code == 200


def test_invalid_modality():
    response = client.post("/reconstruct", json={"modality": "ultrasound"})
    assert response.status_code == 422

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_rejects_empty_question():
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400

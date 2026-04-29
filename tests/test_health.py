"""Health and startup tests."""


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_list_documents(client):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_session_create_and_delete(client):
    # create
    r = client.post("/session")
    assert r.status_code == 200
    session_id = r.json()["session_id"]
    assert session_id

    # get history
    r = client.get(f"/session/{session_id}/history")
    assert r.status_code == 200
    assert r.json()["history"] == []

    # delete
    r = client.delete(f"/session/{session_id}")
    assert r.status_code == 200

    # 404 after delete
    r = client.get(f"/session/{session_id}/history")
    assert r.status_code == 404

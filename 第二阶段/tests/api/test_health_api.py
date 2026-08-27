def test_health_and_openapi(api_context) -> None:
    response = api_context.client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    schema = api_context.client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/sessions" in paths
    assert "/sessions/{session_id}/documents" in paths
    assert "/sessions/{session_id}/questions" in paths


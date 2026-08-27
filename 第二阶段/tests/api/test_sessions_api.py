from 第二阶段.tests.api.conftest import create_session


def test_create_session_has_unique_id_and_timestamp(api_context) -> None:
    first = api_context.client.post("/sessions")
    second = api_context.client.post("/sessions")
    assert first.status_code == second.status_code == 201
    assert first.json()["session_id"] != second.json()["session_id"]
    assert first.json()["created_at"]


def test_delete_session_then_access_returns_404(api_context) -> None:
    session_id = create_session(api_context.client)
    deleted = api_context.client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "session_id": session_id}
    missing = api_context.client.get(f"/sessions/{session_id}/documents")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Session not found."}


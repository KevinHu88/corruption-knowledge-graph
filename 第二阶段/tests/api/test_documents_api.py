from 第二阶段.tests.api.conftest import create_session


def test_upload_txt_and_list_without_chunk_content(api_context) -> None:
    session_id = create_session(api_context.client)
    response = api_context.client.post(
        f"/sessions/{session_id}/documents",
        files={"file": ("evidence.txt", "项目材料包含审批证据。", "text/plain")},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["file_name"] == "evidence.txt"
    assert payload["file_type"] == "txt"
    assert payload["chunk_count"] == 1
    assert payload["status"] == "ready"

    listed = api_context.client.get(f"/sessions/{session_id}/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"] == [payload]
    assert "content" not in listed.text


def test_invalid_pdf_returns_400(api_context) -> None:
    session_id = create_session(api_context.client)
    response = api_context.client.post(
        f"/sessions/{session_id}/documents",
        files={"file": ("broken.pdf", b"not-a-pdf", "application/pdf")},
    )
    assert response.status_code == 400


def test_unsupported_file_returns_415(api_context) -> None:
    session_id = create_session(api_context.client)
    response = api_context.client.post(
        f"/sessions/{session_id}/documents",
        files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_file_too_large_returns_413(api_context) -> None:
    session_id = create_session(api_context.client)
    response = api_context.client.post(
        f"/sessions/{session_id}/documents",
        files={"file": ("large.txt", b"x" * 257, "text/plain")},
    )
    assert response.status_code == 413


def test_session_document_and_retrieval_isolation(api_context) -> None:
    session_a = create_session(api_context.client)
    session_b = create_session(api_context.client)
    uploaded = api_context.client.post(
        f"/sessions/{session_a}/documents",
        files={
            "file": (
                "apple.txt",
                "APPLE_ONLY_INFORMATION belongs to session A.",
                "text/plain",
            )
        },
    )
    assert uploaded.status_code == 201
    documents_b = api_context.client.get(f"/sessions/{session_b}/documents")
    assert documents_b.json()["documents"] == []
    chunks_b = api_context.container.session_service.get_session(
        session_b
    ).document_store.get_chunks()
    assert chunks_b == []
    question_b = api_context.client.post(
        f"/sessions/{session_b}/questions",
        json={"question": "上传文件中的 APPLE_ONLY_INFORMATION 是什么？"},
    )
    assert question_b.status_code == 200
    assert all(
        "APPLE_ONLY_INFORMATION" not in item["content"]
        for item in question_b.json()["evidence"]
    )

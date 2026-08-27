from 第二阶段.routing.query_router import QueryRouter


def test_graph_route_without_uploaded_documents() -> None:
    plan = QueryRouter().route("张某与哪些人存在关系？", False)
    assert plan.route == "GRAPH"


def test_document_route_for_uploaded_material() -> None:
    plan = QueryRouter().route("上传材料中如何描述这个项目？", True)
    assert plan.route == "DOCUMENT"


def test_hybrid_route_for_relation_and_source_evidence() -> None:
    plan = QueryRouter().route("张某与李某有什么关系，原文证据是什么？", True)
    assert plan.route == "HYBRID"


"""集中管理适用于第一阶段 Claim 中心图模型的只读 Cypher。"""

from __future__ import annotations

from typing import Any, Protocol


class ReadOnlyGraphAdapter(Protocol):
    def find_entities(self, **filters: Any) -> list[dict[str, Any]]: ...

    def execute_read(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        *,
        max_records: int = 100,
    ) -> list[dict[str, Any]]: ...


class GraphRepository:
    """实体、Claim、邻居和简单路径的唯一查询入口。"""

    def __init__(self, adapter: ReadOnlyGraphAdapter) -> None:
        self.adapter = adapter

    def find_entity_by_name(
        self,
        name: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        exact = self.adapter.find_entities(name=name, case_id=case_id, limit=limit)
        if exact:
            return [
                {"entity": record.get("e", record.get("entity", record))}
                for record in exact
            ]
        cypher = (
            "MATCH (e:Entity) "
            "WHERE (e.name CONTAINS $name OR e.normalized_name CONTAINS $name) "
            "AND ($case_id IS NULL OR e.case_id = $case_id) "
            "RETURN e AS entity ORDER BY e.name LIMIT $limit"
        )
        return self.adapter.execute_read(
            cypher,
            {"name": name, "case_id": case_id, "limit": limit},
            max_records=limit,
        )

    def find_entities_in_text(
        self,
        text: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.name IS NOT NULL AND size(e.name) >= 2 "
            "AND $text CONTAINS e.name "
            "AND ($case_id IS NULL OR e.case_id = $case_id) "
            "RETURN e AS entity ORDER BY size(e.name) DESC LIMIT $limit"
        )
        return self.adapter.execute_read(
            cypher,
            {"text": text, "case_id": case_id, "limit": limit},
            max_records=limit,
        )

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        cypher = (
            "MATCH (claim:Claim)-[:HEAD]->(head:Entity) "
            "MATCH (claim)-[:TAIL]->(tail:Entity) "
            "WHERE (head.entity_uid = $entity_uid OR tail.entity_uid = $entity_uid) "
            "AND claim.status <> 'REJECTED' "
            "AND ($case_id IS NULL OR claim.case_id = $case_id) "
            "OPTIONAL MATCH (claim)-[:SUPPORTED_BY]->(evidence:TextSpan) "
            "OPTIONAL MATCH (evidence)-[:FROM_DOCUMENT]->(document:SourceDocument) "
            "OPTIONAL MATCH (document)-[:BELONGS_TO_CASE]->(case_node:Case) "
            "RETURN claim, head, tail, collect(DISTINCT evidence) AS evidence, "
            "document, case_node AS case "
            "ORDER BY claim.status, claim.claim_id LIMIT $limit"
        )
        return self.adapter.execute_read(
            cypher,
            {"entity_uid": entity_uid, "case_id": case_id, "limit": limit},
            max_records=limit,
        )

    def get_neighbors(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.get_one_hop_subgraph(entity_uid, limit, case_id=case_id)

    def get_relationships(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.get_one_hop_subgraph(entity_uid, limit, case_id=case_id)

    def find_simple_paths(
        self,
        start_uid: str,
        end_uid: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]:
        """返回两个实体间有界、无重复实体的多条 Claim 路径。"""
        self._validate_path_parameters(max_hops=max_hops, limit=limit)
        relationship_depth = max_hops * 2
        cypher = (
            "MATCH path=(start:Entity {entity_uid: $start_uid})"
            f"-[:HEAD|TAIL*2..{relationship_depth}]-"
            "(end:Entity {entity_uid: $end_uid}) "
            "WITH path, "
            "[node IN nodes(path) WHERE node:Entity | node] AS path_entities, "
            "[node IN nodes(path) WHERE node:Claim | node] AS path_claims "
            "WHERE all(claim IN path_claims WHERE claim.status <> 'REJECTED') "
            "AND ($case_id IS NULL OR all(claim IN path_claims "
            "WHERE claim.case_id = $case_id)) "
            "AND all(entity IN path_entities WHERE single(other IN path_entities "
            "WHERE other = entity)) "
            "RETURN path_entities, path_claims, "
            "[relationship IN relationships(path) | "
            "type(relationship)] AS path_relationship_types, "
            "size(path_claims) AS hop_count "
            "ORDER BY hop_count, "
            "reduce(signature = '', claim IN path_claims | "
            "signature + coalesce(claim.claim_id, '')) "
            "LIMIT $limit"
        )
        return self.adapter.execute_read(
            cypher,
            {
                "start_uid": start_uid,
                "end_uid": end_uid,
                "case_id": case_id,
                "limit": limit,
            },
            max_records=limit,
        )

    def find_path_candidates(
        self,
        relation_types: list[str],
        *,
        start_entity_type: str | None = None,
        end_entity_type: str | None = None,
        exclude_claim_ids: list[str] | None = None,
        case_id: str | None = None,
        case_ids: list[str] | None = None,
        max_hops: int = 3,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """为路径相似度重排读取结构相关的有界候选路径。"""
        self._validate_path_parameters(max_hops=max_hops, limit=limit)
        if not relation_types:
            return []
        scoped_case_ids = list(
            dict.fromkeys(case_ids or ([case_id] if case_id else []))
        )
        relationship_depth = max_hops * 2
        cypher = (
            "MATCH path=(start:Entity)"
            f"-[:HEAD|TAIL*2..{relationship_depth}]-"
            "(end:Entity) "
            "WHERE start.entity_uid < end.entity_uid "
            "AND ($start_entity_type IS NULL OR $end_entity_type IS NULL OR "
            "(start.entity_type = $start_entity_type "
            "AND end.entity_type = $end_entity_type) OR "
            "(start.entity_type = $end_entity_type "
            "AND end.entity_type = $start_entity_type)) "
            "WITH path, "
            "[node IN nodes(path) WHERE node:Entity | node] AS path_entities, "
            "[node IN nodes(path) WHERE node:Claim | node] AS path_claims "
            "WHERE all(claim IN path_claims WHERE claim.status <> 'REJECTED') "
            "AND any(claim IN path_claims "
            "WHERE claim.relation_type IN $relation_types) "
            "AND none(claim IN path_claims "
            "WHERE claim.claim_id IN $exclude_claim_ids) "
            "AND (size($case_ids) = 0 OR all(claim IN path_claims "
            "WHERE claim.case_id IN $case_ids)) "
            "AND size(reduce(case_ids = [], claim IN path_claims | "
            "CASE WHEN claim.case_id IS NULL OR claim.case_id IN case_ids "
            "THEN case_ids ELSE case_ids + claim.case_id END)) <= 1 "
            "AND all(entity IN path_entities WHERE single(other IN path_entities "
            "WHERE other = entity)) "
            "RETURN path_entities, path_claims, "
            "[relationship IN relationships(path) | "
            "type(relationship)] AS path_relationship_types, "
            "size(path_claims) AS hop_count "
            "ORDER BY hop_count, "
            "reduce(signature = '', claim IN path_claims | "
            "signature + coalesce(claim.claim_id, '')) "
            "LIMIT $limit"
        )
        return self.adapter.execute_read(
            cypher,
            {
                "relation_types": relation_types,
                "start_entity_type": start_entity_type,
                "end_entity_type": end_entity_type,
                "exclude_claim_ids": list(exclude_claim_ids or []),
                "case_ids": scoped_case_ids,
                "limit": limit,
            },
            max_records=limit,
        )

    @staticmethod
    def _validate_path_parameters(*, max_hops: int, limit: int) -> None:
        if not 1 <= max_hops <= 5:
            raise ValueError("max_hops 必须位于 1..5")
        if limit <= 0:
            raise ValueError("limit 必须大于 0")

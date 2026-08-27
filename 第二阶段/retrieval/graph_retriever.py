"""把第一阶段实体与 Claim 查询结果转换成统一 Evidence。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Protocol

from 第二阶段.exceptions import AmbiguousEntityError
from 第二阶段.retrieval.path_similarity import (
    PathSignature,
    PathSimilarityScorer,
)
from 第二阶段.schemas.models import Evidence
from 第二阶段.schemas.models import PathSearchScope


class GraphRepositoryProtocol(Protocol):
    def find_entities_in_text(
        self,
        text: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def find_entity_by_name(
        self,
        name: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_one_hop_subgraph(
        self,
        entity_uid: str,
        limit: int = 20,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def find_simple_paths(
        self,
        start_uid: str,
        end_uid: str,
        limit: int = 10,
        *,
        case_id: str | None = None,
        max_hops: int = 3,
    ) -> list[dict[str, Any]]: ...

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
    ) -> list[dict[str, Any]]: ...


def _properties(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get("properties")
    return dict(nested) if isinstance(nested, dict) else dict(value)


@dataclass(slots=True)
class _GraphPath:
    record: dict[str, Any]
    entities: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    directions: list[str]

    @property
    def signature(self) -> PathSignature:
        return PathSignature(
            relation_types=tuple(
                str(claim.get("relation_type") or "存在关系")
                for claim in self.claims
            ),
            entity_types=tuple(
                str(entity.get("entity_type") or "UNKNOWN")
                for entity in self.entities
            ),
            directions=tuple(self.directions),
        )

    @property
    def claim_ids(self) -> list[str]:
        return [
            str(claim.get("claim_id"))
            for claim in self.claims
            if claim.get("claim_id")
        ]

    @property
    def entity_uids(self) -> list[str]:
        return [
            str(entity.get("entity_uid") or entity.get("name") or "")
            for entity in self.entities
        ]

    @property
    def key(self) -> str:
        tokens = self.claim_ids or self.entity_uids
        forward = "|".join(tokens)
        reverse = "|".join(reversed(tokens))
        return min(forward, reverse)


class GraphRetriever:
    PATH_TERMS = ("路径", "关系链", "链路")
    SIMILAR_PATH_TERMS = (
        "相似路径",
        "类似路径",
        "路径相似",
        "相似关系链",
        "类似关系链",
    )

    def __init__(
        self,
        repository: GraphRepositoryProtocol,
        *,
        max_path_hops: int = 3,
        path_candidate_limit: int = 100,
        path_similarity_threshold: float = 0.55,
        path_similarity_scorer: PathSimilarityScorer | None = None,
    ) -> None:
        if not 1 <= max_path_hops <= 5:
            raise ValueError("max_path_hops 必须位于 1..5")
        if path_candidate_limit <= 0:
            raise ValueError("path_candidate_limit 必须大于 0")
        if not 0.0 <= path_similarity_threshold <= 1.0:
            raise ValueError("path_similarity_threshold 必须位于 0..1")
        self.repository = repository
        self.max_path_hops = max_path_hops
        self.path_candidate_limit = path_candidate_limit
        self.path_similarity_threshold = path_similarity_threshold
        self.path_similarity_scorer = (
            path_similarity_scorer or PathSimilarityScorer()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        case_id: str | None = None,
        search_scope: PathSearchScope = "same_case",
        selected_case_ids: list[str] | None = None,
    ) -> list[Evidence]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        normalized_selected_case_ids = self._validate_search_scope(
            search_scope,
            case_id=case_id,
            selected_case_ids=selected_case_ids,
        )
        entities = self.repository.find_entities_in_text(
            query,
            limit=top_k,
            case_id=case_id,
        )
        if not entities:
            for candidate in self._fallback_candidates(query):
                entities.extend(
                    self.repository.find_entity_by_name(
                        candidate,
                        limit=3,
                        case_id=case_id,
                    )
                )
        if case_id:
            entities = [
                record
                for record in entities
                if self._matches_case(record, case_id)
            ]
        normalized_entities = self._deduplicate_entities(entities)
        if not case_id:
            self._raise_for_ambiguous_entities(normalized_entities)
        if self._is_path_query(query) and len(normalized_entities) >= 2:
            path_evidence = self._retrieve_path_evidence(
                normalized_entities,
                query=query,
                top_k=top_k,
                case_id=case_id,
                search_scope=search_scope,
                selected_case_ids=normalized_selected_case_ids,
            )
            if path_evidence:
                path_evidence.sort(
                    key=lambda item: (-(item.score or 0.0), item.id)
                )
                return path_evidence[:top_k]
        evidence: list[Evidence] = []
        seen_claims: set[str] = set()
        for record in normalized_entities:
            entity = record.get("entity", record)
            entity_props = _properties(entity)
            entity_uid = str(entity_props.get("entity_uid") or "")
            if not entity_uid:
                continue
            relations = self.repository.get_one_hop_subgraph(
                entity_uid,
                limit=top_k,
                case_id=case_id,
            )
            if case_id:
                relations = [
                    relation
                    for relation in relations
                    if self._matches_case(relation, case_id)
                ]
            else:
                relation_case_ids = self._case_ids(record)
                for relation in relations:
                    relation_case_ids.update(self._case_ids(relation))
                if len(relation_case_ids) > 1:
                    name = self._entity_name(entity_props)
                    raise AmbiguousEntityError(name, sorted(relation_case_ids))

            evidence.append(self._entity_evidence(entity_props))
            for relation in relations:
                claim_props = _properties(relation.get("claim"))
                claim_id = str(claim_props.get("claim_id") or "")
                if not claim_id or claim_id in seen_claims:
                    continue
                seen_claims.add(claim_id)
                evidence.append(self._claim_evidence(relation, claim_props))
        evidence.sort(key=lambda item: (-(item.score or 0.0), item.id))
        return evidence[:top_k]

    def _retrieve_path_evidence(
        self,
        entity_records: list[dict[str, Any]],
        *,
        query: str,
        top_k: int,
        case_id: str | None,
        search_scope: PathSearchScope,
        selected_case_ids: list[str],
    ) -> list[Evidence]:
        path_finder = getattr(self.repository, "find_simple_paths", None)
        if not callable(path_finder):
            return []
        anchors: list[_GraphPath] = []
        seen_paths: set[str] = set()
        pair_limit = max(top_k * 2, top_k)
        for pair_index, (left_record, right_record) in enumerate(
            combinations(entity_records, 2)
        ):
            if pair_index >= top_k:
                break
            left = _properties(left_record.get("entity", left_record))
            right = _properties(right_record.get("entity", right_record))
            start_uid = str(left.get("entity_uid") or "")
            end_uid = str(right.get("entity_uid") or "")
            if not start_uid or not end_uid:
                continue
            records = path_finder(
                start_uid,
                end_uid,
                limit=pair_limit,
                case_id=case_id,
                max_hops=self.max_path_hops,
            )
            for record in records:
                path = self._parse_path(record)
                if path is None or path.key in seen_paths:
                    continue
                self._validate_path_case(path, case_id=case_id)
                seen_paths.add(path.key)
                anchors.append(path)
        if not anchors:
            return []

        result = [self._path_evidence(path) for path in anchors]
        if self._is_similar_path_query(query):
            result.extend(
                self._similar_path_evidence(
                    anchors,
                    top_k=top_k,
                    case_id=case_id,
                    seen_paths=seen_paths,
                    search_scope=search_scope,
                    selected_case_ids=selected_case_ids,
                )
            )
        return result

    def _similar_path_evidence(
        self,
        anchors: list[_GraphPath],
        *,
        top_k: int,
        case_id: str | None,
        seen_paths: set[str],
        search_scope: PathSearchScope,
        selected_case_ids: list[str],
    ) -> list[Evidence]:
        candidate_finder = getattr(self.repository, "find_path_candidates", None)
        if not callable(candidate_finder):
            return []
        best_by_path: dict[str, Evidence] = {}
        for anchor in anchors:
            anchor_case_ids = self._path_case_ids(anchor)
            if len(anchor_case_ids) > 1:
                raise AmbiguousEntityError(
                    self._path_entity_name(anchor),
                    sorted(anchor_case_ids),
                )
            candidate_case_id = case_id or next(iter(anchor_case_ids), None)
            candidate_case_ids = self._candidate_case_ids(
                search_scope,
                anchor_case_id=candidate_case_id,
                selected_case_ids=selected_case_ids,
            )
            signature = anchor.signature
            records = candidate_finder(
                list(signature.relation_types),
                start_entity_type=signature.entity_types[0],
                end_entity_type=signature.entity_types[-1],
                exclude_claim_ids=anchor.claim_ids,
                case_ids=candidate_case_ids,
                max_hops=self.max_path_hops,
                limit=self.path_candidate_limit,
            )
            for record in records:
                candidate = self._parse_path(record)
                if candidate is None or candidate.key in seen_paths:
                    continue
                if not self._matches_candidate_scope(
                    candidate,
                    allowed_case_ids=candidate_case_ids,
                ):
                    continue
                similarity = self.path_similarity_scorer.score(
                    signature,
                    candidate.signature,
                )
                if similarity.score < self.path_similarity_threshold:
                    continue
                evidence = self._path_evidence(
                    candidate,
                    kind="similar_path",
                    score=similarity.score,
                    extra_metadata={
                        "anchor_path_id": self._path_id(anchor),
                        "similarity": similarity.as_dict(),
                        "search_scope": search_scope,
                        "candidate_case_id": next(
                            iter(self._path_case_ids(candidate)),
                            None,
                        ),
                    },
                )
                previous = best_by_path.get(candidate.key)
                if previous is None or (evidence.score or 0.0) > (
                    previous.score or 0.0
                ):
                    best_by_path[candidate.key] = evidence
        ranked = sorted(
            best_by_path.values(),
            key=lambda item: (-(item.score or 0.0), item.id),
        )
        return ranked[:top_k]

    @staticmethod
    def _validate_search_scope(
        search_scope: PathSearchScope,
        *,
        case_id: str | None,
        selected_case_ids: list[str] | None,
    ) -> list[str]:
        if search_scope not in {"same_case", "selected_cases", "all_cases"}:
            raise ValueError("search_scope 不受支持")
        normalized = list(
            dict.fromkeys(
                value.strip()
                for value in selected_case_ids or []
                if value.strip()
            )
        )
        if search_scope != "same_case" and not case_id:
            raise ValueError("跨案件相似检索必须提供锚点 case_id")
        if search_scope == "selected_cases" and not normalized:
            raise ValueError("selected_cases 必须提供 selected_case_ids")
        if search_scope != "selected_cases" and normalized:
            raise ValueError("selected_case_ids 仅适用于 selected_cases")
        return normalized

    @staticmethod
    def _candidate_case_ids(
        search_scope: PathSearchScope,
        *,
        anchor_case_id: str | None,
        selected_case_ids: list[str],
    ) -> list[str]:
        if search_scope == "all_cases":
            return []
        if search_scope == "selected_cases":
            return selected_case_ids
        return [anchor_case_id] if anchor_case_id else []

    @classmethod
    def _matches_candidate_scope(
        cls,
        path: _GraphPath,
        *,
        allowed_case_ids: list[str],
    ) -> bool:
        path_case_ids = cls._path_case_ids(path)
        if len(path_case_ids) != 1:
            return False
        return not allowed_case_ids or bool(
            path_case_ids.intersection(allowed_case_ids)
        )

    @classmethod
    def _parse_path(cls, record: dict[str, Any]) -> _GraphPath | None:
        raw_entities = record.get("path_entities") or record.get("entities") or []
        raw_claims = record.get("path_claims") or record.get("claims") or []
        entities = [_properties(item) for item in raw_entities]
        claims = [_properties(item) for item in raw_claims]
        if not claims or len(entities) != len(claims) + 1:
            return None
        raw_directions = record.get("directions") or []
        directions = [str(item).lower() for item in raw_directions]
        if len(directions) != len(claims):
            relationship_types = [
                str(item).upper()
                for item in record.get("path_relationship_types") or []
            ]
            if not relationship_types:
                relationships = record.get("path_relationships") or []
                relationship_types = [
                    str(
                        item.get("type")
                        or _properties(item).get("type")
                        or ""
                    ).upper()
                    for item in relationships
                    if isinstance(item, dict)
                ]
            directions = []
            for index in range(len(claims)):
                pair = relationship_types[index * 2 : index * 2 + 2]
                directions.append(
                    "reverse" if pair == ["TAIL", "HEAD"] else "forward"
                )
        return _GraphPath(record, entities, claims, directions)

    @classmethod
    def _path_evidence(
        cls,
        path: _GraphPath,
        *,
        kind: str = "path",
        score: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        path_id = cls._path_id(path)
        chain = cls._path_chain(path)
        quality_score = sum(
            cls._claim_status_score(claim) for claim in path.claims
        ) / len(path.claims)
        actual_score = score
        if actual_score is None:
            actual_score = quality_score * (1.0 - 0.04 * (len(path.claims) - 1))
        case_ids = sorted(cls._path_case_ids(path))
        doc_ids = list(
            dict.fromkeys(
                str(claim.get("doc_id"))
                for claim in path.claims
                if claim.get("doc_id")
            )
        )
        metadata: dict[str, Any] = {
            "kind": kind,
            "path_id": path_id,
            "hop_count": len(path.claims),
            "path_entities": path.entities,
            "path_claims": path.claims,
            "directions": path.directions,
            "relation_signature": list(path.signature.relation_types),
            "entity_type_signature": list(path.signature.entity_types),
            "claim_ids": path.claim_ids,
            "case_ids": case_ids,
            "document_ids": doc_ids,
            "path_quality_score": round(quality_score, 6),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        prefix = "相似路径" if kind == "similar_path" else "关系路径"
        return Evidence(
            id=f"graph-{kind}-{path_id}",
            source_type="graph",
            content=f"{prefix}（{len(path.claims)}跳）：{chain}",
            score=round(actual_score, 6),
            source=case_ids[0] if len(case_ids) == 1 else "Neo4j",
            metadata=metadata,
        )

    @staticmethod
    def _path_chain(path: _GraphPath) -> str:
        names = [
            str(
                entity.get("name")
                or entity.get("normalized_name")
                or entity.get("entity_uid")
                or "未知实体"
            )
            for entity in path.entities
        ]
        pieces = [names[0]]
        for index, claim in enumerate(path.claims):
            relation_type = str(claim.get("relation_type") or "存在关系")
            if path.directions[index] == "reverse":
                pieces.append(f" <-[{relation_type}]- {names[index + 1]}")
            else:
                pieces.append(f" -[{relation_type}]-> {names[index + 1]}")
        return "".join(pieces) + "。"

    @staticmethod
    def _claim_status_score(claim: dict[str, Any]) -> float:
        status = str(claim.get("status") or "")
        if status == "HUMAN_VERIFIED":
            return 1.0
        if status == "MODEL_PREDICTED":
            return 0.85
        return 0.75

    @staticmethod
    def _path_id(path: _GraphPath) -> str:
        digest = hashlib.sha1(path.key.encode("utf-8")).hexdigest()
        return digest[:16]

    @staticmethod
    def _path_case_ids(path: _GraphPath) -> set[str]:
        values = {
            str(item.get("case_id"))
            for item in [*path.entities, *path.claims]
            if item.get("case_id")
        }
        return values

    @classmethod
    def _validate_path_case(
        cls,
        path: _GraphPath,
        *,
        case_id: str | None,
    ) -> None:
        path_case_ids = cls._path_case_ids(path)
        if case_id and path_case_ids and path_case_ids != {case_id}:
            raise AmbiguousEntityError(
                cls._path_entity_name(path),
                sorted(path_case_ids),
            )
        if not case_id and len(path_case_ids) > 1:
            raise AmbiguousEntityError(
                cls._path_entity_name(path),
                sorted(path_case_ids),
            )

    @staticmethod
    def _path_entity_name(path: _GraphPath) -> str:
        entity = path.entities[0]
        return str(
            entity.get("name")
            or entity.get("normalized_name")
            or entity.get("entity_uid")
            or "未知实体"
        )

    @classmethod
    def _is_path_query(cls, query: str) -> bool:
        return any(term in query for term in cls.PATH_TERMS)

    @classmethod
    def _is_similar_path_query(cls, query: str) -> bool:
        return any(term in query for term in cls.SIMILAR_PATH_TERMS)

    @classmethod
    def _raise_for_ambiguous_entities(
        cls, records: list[dict[str, Any]]
    ) -> None:
        cases_by_name: dict[str, set[str]] = {}
        for record in records:
            props = _properties(record.get("entity", record))
            name = cls._entity_name(props)
            cases_by_name.setdefault(name, set()).update(cls._case_ids(record))
        for name, case_ids in cases_by_name.items():
            if len(case_ids) > 1:
                raise AmbiguousEntityError(name, sorted(case_ids))

    @staticmethod
    def _entity_name(properties: dict[str, Any]) -> str:
        return str(
            properties.get("normalized_name")
            or properties.get("name")
            or properties.get("entity_uid")
            or "未知实体"
        )

    @staticmethod
    def _case_ids(record: dict[str, Any]) -> set[str]:
        values: set[str] = set()
        for key in ("entity", "claim", "case", "head", "tail"):
            properties = _properties(record.get(key))
            value = str(properties.get("case_id") or "").strip()
            if value:
                values.add(value)
        if not any(key in record for key in ("entity", "claim", "case", "head", "tail")):
            value = str(_properties(record).get("case_id") or "").strip()
            if value:
                values.add(value)
        return values

    @classmethod
    def _matches_case(cls, record: dict[str, Any], case_id: str) -> bool:
        record_case_ids = cls._case_ids(record)
        return not record_case_ids or record_case_ids == {case_id}

    @staticmethod
    def _fallback_candidates(query: str) -> list[str]:
        candidates = re.findall(r"[\u4e00-\u9fff]{1,4}某", query)
        candidates.extend(re.findall(r"[“\"]([^”\"]{2,30})[”\"]", query))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _deduplicate_entities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            props = _properties(record.get("entity", record))
            key = str(props.get("entity_uid") or props.get("name") or record)
            if key not in seen:
                seen.add(key)
                result.append(record)
        return result

    @staticmethod
    def _entity_evidence(properties: dict[str, Any]) -> Evidence:
        uid = str(properties.get("entity_uid") or properties.get("name"))
        name = str(properties.get("name") or properties.get("normalized_name") or uid)
        entity_type = str(properties.get("entity_type") or "未知类型")
        return Evidence(
            id=f"graph-entity-{uid}",
            source_type="graph",
            content=f"知识图谱实体：{name}（类型：{entity_type}）。",
            score=0.6,
            source=str(properties.get("case_id") or "Neo4j"),
            metadata={"entity": properties},
        )

    @staticmethod
    def _claim_evidence(record: dict[str, Any], claim: dict[str, Any]) -> Evidence:
        head = _properties(record.get("head"))
        tail = _properties(record.get("tail"))
        relation_type = str(claim.get("relation_type") or "存在关系")
        head_name = str(head.get("name") or head.get("normalized_name") or "未知实体")
        tail_name = str(tail.get("name") or tail.get("normalized_name") or "未知实体")
        evidence_nodes = record.get("evidence") or []
        spans = [_properties(item) for item in evidence_nodes if isinstance(item, dict)]
        evidence_text = str(
            claim.get("evidence_text")
            or next((item.get("text") for item in spans if item.get("text")), "")
        )
        content = f"{head_name}与{tail_name}之间存在“{relation_type}”关系。"
        if evidence_text:
            content += f" 原始证据：{evidence_text}"
        document = _properties(record.get("document"))
        case = _properties(record.get("case"))
        source = (
            claim.get("source_url")
            or document.get("canonical_url")
            or document.get("raw_url")
            or document.get("title")
            or claim.get("doc_id")
            or case.get("case_id")
            or "Neo4j"
        )
        status_score = 1.0 if claim.get("status") == "HUMAN_VERIFIED" else 0.85
        return Evidence(
            id=f"graph-claim-{claim.get('claim_id')}",
            source_type="graph",
            content=content,
            score=status_score,
            source=str(source),
            metadata={
                "start_node": head,
                "relationship": relation_type,
                "end_node": tail,
                "relationship_properties": claim,
                "node_properties": {"head": head, "tail": tail},
                "evidence": spans,
                "source_document": document,
                "case": case,
            },
        )

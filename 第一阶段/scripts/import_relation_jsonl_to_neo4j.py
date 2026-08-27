"""Import OpenNRE-style relation JSONL files into the stage-one graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, load_project_config
from models import (
    AnnotationStatus,
    CanonicalAnnotation,
    EntityMention,
    EntityType,
    RelationMention,
    RelationType,
)
from src.services.neo4j_service import Neo4jService


DEFAULT_INPUTS = (
    BASE_DIR / "artifacts/datasets/mydata-v1/relation/train.jsonl",
    BASE_DIR / "artifacts/datasets/mydata-v1/relation/validation.jsonl",
)
RELATION_ALIASES = {"关联人物": "职务制约"}


@dataclass
class RelationRecord:
    text: str
    head: dict[str, Any]
    tail: dict[str, Any]
    relation: str
    original_relation: str
    sources: list[str] = field(default_factory=list)


@dataclass
class ConversionStats:
    input_rows: int = 0
    duplicate_rows: int = 0
    repaired_offsets: int = 0
    normalized_relations: int = 0


def _digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _repair_entity(
    text: str,
    raw_entity: dict[str, Any],
    *,
    source: str,
    stats: ConversionStats,
) -> dict[str, Any]:
    entity = dict(raw_entity)
    name = str(entity["name"])
    start, end = (int(value) for value in entity["pos"])
    if text[start:end] == name:
        return entity

    corrected_end = start + len(name)
    if text[start:corrected_end] == name:
        entity["pos"] = [start, corrected_end]
        stats.repaired_offsets += 1
        return entity

    nearby_start = max(0, start - 8)
    nearby_end = min(len(text), end + 8)
    matches = []
    cursor = nearby_start
    while True:
        match = text.find(name, cursor, nearby_end)
        if match < 0:
            break
        matches.append(match)
        cursor = match + 1
    if len(matches) == 1:
        entity["pos"] = [matches[0], matches[0] + len(name)]
        stats.repaired_offsets += 1
        return entity
    raise ValueError(
        f"{source}: entity offset is invalid and cannot be repaired: "
        f"{name!r} at [{start}, {end}]"
    )


def load_records(paths: list[Path]) -> tuple[list[RelationRecord], ConversionStats]:
    stats = ConversionStats()
    records_by_key: dict[tuple[Any, ...], RelationRecord] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                stats.input_rows += 1
                raw = json.loads(line)
                source = f"{path.name}:{line_number}"
                text = str(raw["text"])
                head = _repair_entity(
                    text, raw["h"], source=source, stats=stats
                )
                tail = _repair_entity(
                    text, raw["t"], source=source, stats=stats
                )
                original_relation = str(raw["relation"])
                relation = RELATION_ALIASES.get(
                    original_relation, original_relation
                )
                if relation != original_relation:
                    stats.normalized_relations += 1
                key = (
                    text,
                    head["name"],
                    head["type"],
                    tuple(head["pos"]),
                    tail["name"],
                    tail["type"],
                    tuple(tail["pos"]),
                    relation,
                )
                if key in records_by_key:
                    records_by_key[key].sources.append(source)
                    stats.duplicate_rows += 1
                    continue
                records_by_key[key] = RelationRecord(
                    text=text,
                    head=head,
                    tail=tail,
                    relation=relation,
                    original_relation=original_relation,
                    sources=[source],
                )
    return list(records_by_key.values()), stats


def build_annotations(
    records: list[RelationRecord], *, dataset_version: str
) -> list[CanonicalAnnotation]:
    schema = load_project_config().schema_config
    valid_relations = set(schema["relation_types"])
    grouped: dict[str, list[RelationRecord]] = defaultdict(list)
    for record in records:
        if record.relation not in valid_relations:
            raise ValueError(f"relation is not in schema: {record.relation}")
        grouped[record.text].append(record)

    annotations = []
    for text in sorted(grouped, key=lambda value: _digest(value, 64)):
        text_hash = _digest(text)
        chunk_id = f"chunk-{text_hash}"
        entity_specs: dict[tuple[Any, ...], dict[str, Any]] = {}
        for record in grouped[text]:
            for entity in (record.head, record.tail):
                start, end = entity["pos"]
                key = (start, end, entity["name"], entity["type"])
                entity_specs[key] = entity

        entity_ids = {
            key: f"entity-{_digest(json.dumps(key, ensure_ascii=False))}"
            for key in sorted(entity_specs)
        }
        entities = [
            EntityMention(
                entity_id=entity_ids[key],
                name=str(entity_specs[key]["name"]),
                type=EntityType(str(entity_specs[key]["type"])),
                start=int(entity_specs[key]["pos"][0]),
                end=int(entity_specs[key]["pos"][1]),
                normalized_name=str(entity_specs[key]["name"]),
            )
            for key in sorted(entity_specs)
        ]

        relation_items: dict[tuple[str, str, str], RelationMention] = {}
        provenance: dict[str, dict[str, list[str]]] = {}
        for record in grouped[text]:
            head_key = (
                record.head["pos"][0], record.head["pos"][1],
                record.head["name"], record.head["type"],
            )
            tail_key = (
                record.tail["pos"][0], record.tail["pos"][1],
                record.tail["name"], record.tail["type"],
            )
            key = (entity_ids[head_key], entity_ids[tail_key], record.relation)
            relation_id = f"relation-{_digest('|'.join(key))}"
            relation_items[key] = RelationMention(
                relation_id=relation_id,
                head_id=key[0],
                tail_id=key[1],
                type=RelationType(record.relation),
                evidence_start=0,
                evidence_end=len(text),
                extraction_source="HUMAN",
            )
            item = provenance.setdefault(
                relation_id,
                {
                    "dataset_splits": [],
                    "source_files": [],
                    "source_rows": [],
                    "original_relation_types": [],
                },
            )
            item["source_rows"].extend(record.sources)
            item["dataset_splits"].extend(
                source.split(".jsonl:", 1)[0] for source in record.sources
            )
            item["source_files"].extend(
                source.split(":", 1)[0] for source in record.sources
            )
            item["original_relation_types"].append(
                record.original_relation
            )

        for item in provenance.values():
            for name, values in item.items():
                item[name] = sorted(set(values))
        dataset_splits = sorted(
            {
                split
                for item in provenance.values()
                for split in item["dataset_splits"]
            }
        )
        source_files = sorted(
            {
                source
                for item in provenance.values()
                for source in item["source_files"]
            }
        )
        annotations.append(
            CanonicalAnnotation(
                annotation_id=chunk_id,
                case_id=f"case-sim-{text_hash}",
                doc_id=f"doc-sim-{text_hash}",
                text_id=f"text-{text_hash}",
                text=text,
                entities=entities,
                relations=list(relation_items.values()),
                annotation_source="HUMAN",
                schema_version=str(schema["schema_version"]),
                status=AnnotationStatus.APPROVED,
                metadata={
                    "chunk_id": chunk_id,
                    "dataset_version": dataset_version,
                    "dataset_splits": dataset_splits,
                    "source_files": source_files,
                    "source_id": "simulated-relation-dataset",
                    "title": f"{dataset_version} simulated chunk",
                    "doc_version_id": f"doc-version-sim-{text_hash}",
                    "relation_provenance": provenance,
                },
            )
        )
    return annotations


def summarize(
    annotations: list[CanonicalAnnotation], stats: ConversionStats
) -> dict[str, int]:
    return {
        "input_rows": stats.input_rows,
        "duplicate_rows_removed": stats.duplicate_rows,
        "offsets_repaired": stats.repaired_offsets,
        "relation_labels_normalized": stats.normalized_relations,
        "chunks": len(annotations),
        "entities": sum(len(item.entities) for item in annotations),
        "claims": sum(len(item.relations) for item in annotations),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--dataset-version", default="mydata-v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inputs = args.inputs or list(DEFAULT_INPUTS)
    records, stats = load_records(inputs)
    annotations = build_annotations(
        records, dataset_version=args.dataset_version
    )
    output: dict[str, Any] = {"conversion": summarize(annotations, stats)}
    if args.dry_run:
        output["status"] = "dry-run-ok"
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    with Neo4jService() as service:
        schema_result = service.initialize_schema()
        result = service.ingest_annotations_batch(annotations)
        verification = service.execute_read_query(
            "MATCH (c:Claim {dataset_version: $dataset_version}) "
            "WITH count(c) AS claims, count(DISTINCT c.chunk_id) AS chunks "
            "MATCH (s:TextSpan {dataset_version: $dataset_version}) "
            "RETURN claims, chunks, count(DISTINCT s) AS text_spans",
            {"dataset_version": args.dataset_version},
            max_records=1,
        )
    output.update(
        {
            "status": "imported",
            "schema_items": len(schema_result.items),
            "successful_batches": result.successful_batches,
            "failed_batches": result.failed_batches,
            "write_counters": result.counters.model_dump(),
            "verification": verification.records[0]
            if verification.records
            else {},
        }
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

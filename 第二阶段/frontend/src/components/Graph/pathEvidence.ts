import type { Evidence, PathSearchScope } from '@/lib/phase2Api'

export interface PathEntityView {
  id: string
  name: string
  type: string
  caseId?: string
}

export interface PathClaimView {
  id: string
  relationType: string
  status?: string
  evidenceText?: string
  documentId?: string
  sourceUrl?: string
  caseId?: string
}

export interface SimilarityView {
  score: number
  relationSequenceScore: number
  entityTypeSequenceScore: number
  relationOverlapScore: number
  lengthScore: number
  orientation: string
}

export interface GraphPathView {
  evidenceId: string
  kind: 'path' | 'similar_path'
  score: number | null
  source: string | null
  hopCount: number
  entities: PathEntityView[]
  claims: PathClaimView[]
  directions: Array<'forward' | 'reverse'>
  caseIds: string[]
  searchScope?: PathSearchScope
  candidateCaseId?: string
  similarity?: SimilarityView
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function text(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function number(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

function parseSimilarity(value: unknown): SimilarityView | undefined {
  const data = record(value)
  if (!data) return undefined
  return {
    score: number(data.score),
    relationSequenceScore: number(data.relation_sequence_score),
    entityTypeSequenceScore: number(data.entity_type_sequence_score),
    relationOverlapScore: number(data.relation_overlap_score),
    lengthScore: number(data.length_score),
    orientation: text(data.orientation, 'forward'),
  }
}

export function parseGraphPathEvidence(evidence: Evidence): GraphPathView | null {
  const metadata = evidence.metadata
  const kind = metadata.kind
  if (evidence.source_type !== 'graph' || (kind !== 'path' && kind !== 'similar_path')) {
    return null
  }
  const rawEntities = Array.isArray(metadata.path_entities) ? metadata.path_entities : []
  const rawClaims = Array.isArray(metadata.path_claims) ? metadata.path_claims : []
  const entities = rawEntities.flatMap((value, index) => {
    const item = record(value)
    if (!item) return []
    return [{
      id: text(item.entity_uid, `entity-${index}`),
      name: text(item.name) || text(item.normalized_name) || text(item.entity_uid, '未知实体'),
      type: text(item.entity_type, 'UNKNOWN'),
      caseId: text(item.case_id) || undefined,
    }]
  })
  const claims = rawClaims.flatMap((value, index) => {
    const item = record(value)
    if (!item) return []
    return [{
      id: text(item.claim_id, `claim-${index}`),
      relationType: text(item.relation_type, '存在关系'),
      status: text(item.status) || undefined,
      evidenceText: text(item.evidence_text) || undefined,
      documentId: text(item.doc_id) || undefined,
      sourceUrl: text(item.source_url) || undefined,
      caseId: text(item.case_id) || undefined,
    }]
  })
  if (claims.length === 0 || entities.length !== claims.length + 1) return null

  const directions = stringArray(metadata.directions).map((value) => (
    value === 'reverse' ? 'reverse' : 'forward'
  ))
  if (directions.length !== claims.length) return null

  const searchScope = metadata.search_scope
  return {
    evidenceId: evidence.id,
    kind,
    score: evidence.score,
    source: evidence.source,
    hopCount: number(metadata.hop_count, claims.length),
    entities,
    claims,
    directions,
    caseIds: stringArray(metadata.case_ids),
    searchScope: searchScope === 'same_case'
      || searchScope === 'selected_cases'
      || searchScope === 'all_cases'
      ? searchScope
      : undefined,
    candidateCaseId: text(metadata.candidate_case_id) || undefined,
    similarity: parseSimilarity(metadata.similarity),
  }
}

export function splitPathEvidence(evidence: Evidence[]): {
  paths: GraphPathView[]
  remaining: Evidence[]
} {
  const paths: GraphPathView[] = []
  const remaining: Evidence[] = []
  for (const item of evidence) {
    const path = parseGraphPathEvidence(item)
    if (path) paths.push(path)
    else remaining.push(item)
  }
  return { paths, remaining }
}

import type { Evidence } from '@/lib/phase2Api'

import { parseGraphPathEvidence, splitPathEvidence } from './pathEvidence'

const pathEvidence: Evidence = {
  id: 'graph-path-1',
  source_type: 'graph',
  content: '关系路径',
  score: 0.96,
  source: 'case-1',
  metadata: {
    kind: 'similar_path',
    hop_count: 1,
    path_entities: [
      { entity_uid: 'a', name: '张某', entity_type: 'PER', case_id: 'case-2' },
      { entity_uid: 'b', name: '某公司', entity_type: 'ORG', case_id: 'case-2' },
    ],
    path_claims: [
      { claim_id: 'c', relation_type: '任职于', status: 'HUMAN_VERIFIED', evidence_text: '张某任职于某公司。', source_url: 'https://example.test/source' },
    ],
    directions: ['forward'],
    case_ids: ['case-2'],
    search_scope: 'selected_cases',
    candidate_case_id: 'case-2',
    similarity: {
      score: 0.96,
      relation_sequence_score: 1,
      entity_type_sequence_score: 1,
      relation_overlap_score: 1,
      length_score: 1,
      orientation: 'forward',
    },
  },
}

describe('path evidence parsing', () => {
  it('parses graph path metadata for visualization', () => {
    const path = parseGraphPathEvidence(pathEvidence)
    expect(path?.entities.map((item) => item.name)).toEqual(['张某', '某公司'])
    expect(path?.claims[0].relationType).toBe('任职于')
    expect(path?.claims[0].evidenceText).toBe('张某任职于某公司。')
    expect(path?.claims[0].sourceUrl).toBe('https://example.test/source')
    expect(path?.searchScope).toBe('selected_cases')
    expect(path?.candidateCaseId).toBe('case-2')
  })

  it('keeps non-path evidence in the remaining list', () => {
    const documentEvidence: Evidence = {
      ...pathEvidence,
      id: 'document-1',
      source_type: 'document',
      metadata: {},
    }
    const result = splitPathEvidence([pathEvidence, documentEvidence])
    expect(result.paths).toHaveLength(1)
    expect(result.remaining).toEqual([documentEvidence])
  })

  it('rejects malformed paths', () => {
    const malformed = {
      ...pathEvidence,
      metadata: { ...pathEvidence.metadata, directions: [] },
    }
    expect(parseGraphPathEvidence(malformed)).toBeNull()
  })
})

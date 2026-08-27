export interface SessionResponse {
  session_id: string
  created_at: string
}

export interface SessionDocument {
  document_id: string
  file_name: string
  file_type: string
  chunk_count: number
  status: string
}

export interface Evidence {
  id: string
  source_type: 'document' | 'graph'
  content: string
  score: number | null
  source: string | null
  metadata: Record<string, unknown>
}

export interface QuestionResponse {
  answer: string
  route: 'DOCUMENT' | 'GRAPH' | 'HYBRID'
  evidence: Evidence[]
  sources: Array<{
    source_type: 'document' | 'graph'
    name: string
  }>
}

export type PathSearchScope = 'same_case' | 'selected_cases' | 'all_cases'

export interface AskQuestionOptions {
  caseId?: string
  searchScope?: PathSearchScope
  selectedCaseIds?: string[]
}

interface ErrorPayload {
  detail?: string
  entity_name?: string | null
  candidate_case_ids?: string[] | null
}

export class ApiError extends Error {
  status: number
  candidateCaseIds: string[]

  constructor(status: number, payload: ErrorPayload) {
    super(payload.detail || `请求失败（HTTP ${status}）`)
    this.name = 'ApiError'
    this.status = status
    this.candidateCaseIds = payload.candidate_case_ids || []
  }
}

function getApiUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, '')
  }
  if (typeof window !== 'undefined') {
    return `${window.location.protocol}//${window.location.hostname}:8000`
  }
  return 'http://localhost:8000'
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiUrl()}${path}`, options)
  if (!response.ok) {
    let payload: ErrorPayload = {}
    try {
      payload = await response.json()
    } catch {
      payload.detail = response.statusText
    }
    throw new ApiError(response.status, payload)
  }
  return response.json() as Promise<T>
}

export const phase2Api = {
  get baseUrl() {
    return getApiUrl()
  },

  health() {
    return request<{ status: 'ok' }>('/health')
  },

  createSession() {
    return request<SessionResponse>('/sessions', { method: 'POST' })
  },

  deleteSession(sessionId: string) {
    return request<{ deleted: boolean; session_id: string }>(
      `/sessions/${sessionId}`,
      { method: 'DELETE' },
    )
  },

  listDocuments(sessionId: string) {
    return request<{ session_id: string; documents: SessionDocument[] }>(
      `/sessions/${sessionId}/documents`,
    )
  },

  uploadDocument(sessionId: string, file: File) {
    const form = new FormData()
    form.append('file', file)
    return request<SessionDocument>(`/sessions/${sessionId}/documents`, {
      method: 'POST',
      body: form,
    })
  },

  askQuestion(
    sessionId: string,
    question: string,
    options: AskQuestionOptions = {},
  ) {
    return request<QuestionResponse>(`/sessions/${sessionId}/questions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        case_id: options.caseId?.trim() || null,
        search_scope: options.searchScope || 'same_case',
        selected_case_ids: options.selectedCaseIds || [],
      }),
    })
  },
}

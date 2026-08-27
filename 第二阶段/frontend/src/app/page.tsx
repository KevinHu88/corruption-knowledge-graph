'use client'

import {
  ArrowPathIcon,
  ArrowUpTrayIcon,
  BoltIcon,
  CheckCircleIcon,
  CircleStackIcon,
  DocumentTextIcon,
  PaperAirplaneIcon,
  ScaleIcon,
  ShieldCheckIcon,
  UserGroupIcon,
} from '@heroicons/react/24/outline'
import { FormEvent, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import PathEvidencePanel from '@/components/Graph/PathEvidencePanel'
import { splitPathEvidence } from '@/components/Graph/pathEvidence'
import {
  ApiError,
  Evidence,
  PathSearchScope,
  phase2Api,
  QuestionResponse,
  SessionDocument,
} from '@/lib/phase2Api'

interface UiMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  result?: QuestionResponse
}

const routeLabels: Record<QuestionResponse['route'], string> = {
  DOCUMENT: '文档检索',
  GRAPH: '图谱检索',
  HYBRID: '混合检索',
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError && error.candidateCaseIds.length > 0) {
    return `${error.message}。候选案件：${error.candidateCaseIds.join('、')}`
  }
  return error instanceof Error ? error.message : '发生未知错误'
}

function EvidenceCard({ item, index }: { item: Evidence; index: number }) {
  const graph = item.source_type === 'graph'
  return (
    <details className="group rounded-xl border border-slate-200 bg-white/70 p-3 dark:border-slate-700 dark:bg-slate-900/50">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm">
        <span className="flex min-w-0 items-center gap-2 font-medium text-slate-700 dark:text-slate-200">
          <span className={`h-2 w-2 shrink-0 rounded-full ${graph ? 'bg-amber-500' : 'bg-cyan-500'}`} />
          <span className="truncate">证据 {index + 1} · {item.source || (graph ? 'Neo4j 图谱' : '上传文档')}</span>
        </span>
        {item.score !== null && (
          <span className="shrink-0 font-mono text-xs text-slate-400">{item.score.toFixed(3)}</span>
        )}
      </summary>
      <p className="mt-3 whitespace-pre-wrap border-t border-slate-100 pt-3 text-sm leading-6 text-slate-600 dark:border-slate-800 dark:text-slate-300">
        {item.content}
      </p>
    </details>
  )
}

function AssistantEvidence({ evidence }: { evidence: Evidence[] }) {
  const { paths, remaining } = splitPathEvidence(evidence)
  return (
    <>
      {paths.length > 0 && <PathEvidencePanel evidence={evidence} />}
      {remaining.length > 0 && (
        <div className="mt-5 space-y-2 border-t border-slate-100 pt-4 dark:border-slate-800">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">支撑证据</p>
          {remaining.map((item, index) => <EvidenceCard key={`${item.id}-${index}`} item={item} index={index} />)}
        </div>
      )}
    </>
  )
}

export default function Home() {
  const [sessionId, setSessionId] = useState('')
  const [documents, setDocuments] = useState<SessionDocument[]>([])
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [question, setQuestion] = useState('')
  const [caseId, setCaseId] = useState('')
  const [searchScope, setSearchScope] = useState<PathSearchScope>('same_case')
  const [selectedCasesInput, setSelectedCasesInput] = useState('')
  const [allCasesConfirmed, setAllCasesConfirmed] = useState(false)
  const [online, setOnline] = useState(false)
  const [initializing, setInitializing] = useState(true)
  const [asking, setAsking] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const messagesEnd = useRef<HTMLDivElement>(null)

  const createSession = async () => {
    setInitializing(true)
    setNotice(null)
    try {
      await phase2Api.health()
      const session = await phase2Api.createSession()
      setOnline(true)
      setSessionId(session.session_id)
      setDocuments([])
      setMessages([])
    } catch (error) {
      setOnline(false)
      setNotice(`无法连接后端：${errorMessage(error)}`)
    } finally {
      setInitializing(false)
    }
  }

  useEffect(() => {
    void createSession()
  }, [])

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  const resetSession = async () => {
    const previous = sessionId
    await createSession()
    if (previous) {
      try {
        await phase2Api.deleteSession(previous)
      } catch {
        // The new session is usable; stale in-memory cleanup is best effort.
      }
    }
  }

  const upload = async (files: FileList | null) => {
    if (!files?.length || !sessionId) return
    setUploading(true)
    setNotice(null)
    try {
      for (const file of Array.from(files)) {
        await phase2Api.uploadDocument(sessionId, file)
      }
      const result = await phase2Api.listDocuments(sessionId)
      setDocuments(result.documents)
      setNotice(`已上传 ${files.length} 个文件，当前会话共 ${result.documents.length} 个文档。`)
    } catch (error) {
      setNotice(`上传失败：${errorMessage(error)}`)
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const ask = async (event: FormEvent) => {
    event.preventDefault()
    const normalized = question.trim()
    if (!normalized || !sessionId || asking) return
    const selectedCaseIds = selectedCasesInput
      .split(/[\s,，;；]+/)
      .map((value) => value.trim())
      .filter(Boolean)
    if (searchScope !== 'same_case' && !caseId.trim()) {
      setNotice('跨案件相似检索需要先填写锚点案件 ID。')
      return
    }
    if (searchScope === 'selected_cases' && selectedCaseIds.length === 0) {
      setNotice('指定案件检索至少需要填写一个候选案件 ID。')
      return
    }
    if (searchScope === 'all_cases' && !allCasesConfirmed) {
      setNotice('全库跨案检索需要先确认显式授权。')
      return
    }

    setMessages((current) => [...current, {
      id: crypto.randomUUID(),
      role: 'user',
      content: normalized,
    }])
    setQuestion('')
    setAsking(true)
    setNotice(null)

    try {
      const result = await phase2Api.askQuestion(sessionId, normalized, {
        caseId,
        searchScope,
        selectedCaseIds: searchScope === 'selected_cases' ? selectedCaseIds : [],
      })
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.answer,
        result,
      }])
    } catch (error) {
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `请求未完成：${errorMessage(error)}`,
      }])
    } finally {
      setAsking(false)
    }
  }

  return (
    <main className="min-h-screen bg-[#f5f3ee] text-slate-950 dark:bg-slate-950 dark:text-slate-50">
      <div className="mx-auto flex min-h-screen max-w-[1680px] flex-col lg:flex-row">
        <aside className="relative overflow-hidden border-b border-slate-200 bg-[#102a2a] px-5 py-6 text-white lg:w-[360px] lg:shrink-0 lg:border-b-0 lg:border-r lg:border-white/10 lg:px-7 lg:py-8">
          <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl" />
          <div className="relative">
            <div className="mb-9 flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-2xl bg-amber-400 text-[#102a2a] shadow-lg shadow-amber-400/10">
                <ScaleIcon className="h-6 w-6" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-200/70">Integrity Graph</p>
                <h1 className="text-xl font-semibold tracking-tight">腐败知识图谱研判台</h1>
              </div>
            </div>

            <section className="mb-6 rounded-2xl border border-white/10 bg-white/[0.06] p-4 backdrop-blur">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-sm font-medium">服务状态</span>
                <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${online ? 'bg-emerald-400/15 text-emerald-200' : 'bg-rose-400/15 text-rose-200'}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${online ? 'bg-emerald-300' : 'bg-rose-300'}`} />
                  {initializing ? '连接中' : online ? '已连接' : '未连接'}
                </span>
              </div>
              <p className="truncate font-mono text-[11px] text-white/45" title={sessionId}>
                {sessionId ? `SESSION ${sessionId}` : '等待连接第二阶段 API'}
              </p>
              <button type="button" onClick={() => void resetSession()} disabled={initializing} className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-white/10 px-3 py-2.5 text-sm font-medium transition hover:bg-white/15 disabled:opacity-50">
                <ArrowPathIcon className={`h-4 w-4 ${initializing ? 'animate-spin' : ''}`} />
                新建研判会话
              </button>
            </section>

            <section className="mb-6">
              <label htmlFor="case-id" className="mb-2 block text-xs font-semibold uppercase tracking-wider text-white/55">案件范围（可选）</label>
              <input id="case-id" value={caseId} onChange={(event) => setCaseId(event.target.value)} placeholder="例如 CASE-2026-001" className="w-full rounded-xl border border-white/10 bg-black/10 px-3.5 py-3 text-sm outline-none placeholder:text-white/25 focus:border-cyan-300/50 focus:ring-2 focus:ring-cyan-300/10" />
              <p className="mt-2 text-xs leading-5 text-white/40">限定同名实体的案件边界，避免跨案证据混入。</p>
            </section>

            <section className="mb-6 rounded-2xl border border-white/10 bg-black/10 p-4">
              <label htmlFor="search-scope" className="mb-2 block text-xs font-semibold uppercase tracking-wider text-white/55">相似路径范围</label>
              <select
                id="search-scope"
                value={searchScope}
                onChange={(event) => {
                  setSearchScope(event.target.value as PathSearchScope)
                  setAllCasesConfirmed(false)
                }}
                className="w-full rounded-xl border border-white/10 bg-[#102a2a] px-3 py-2.5 text-sm text-white outline-none focus:border-cyan-300/50 focus:ring-2 focus:ring-cyan-300/10"
              >
                <option value="same_case">当前案件（默认）</option>
                <option value="selected_cases">指定多个案件</option>
                <option value="all_cases">全部案件</option>
              </select>
              {searchScope === 'selected_cases' && (
                <div className="mt-3">
                  <label htmlFor="selected-cases" className="mb-1.5 block text-xs text-white/55">候选案件 ID</label>
                  <textarea
                    id="selected-cases"
                    value={selectedCasesInput}
                    onChange={(event) => setSelectedCasesInput(event.target.value)}
                    rows={3}
                    placeholder={'case-001\ncase-002'}
                    className="w-full resize-none rounded-xl border border-white/10 bg-black/10 px-3 py-2.5 font-mono text-xs outline-none placeholder:text-white/25 focus:border-cyan-300/50"
                  />
                </div>
              )}
              {searchScope === 'all_cases' && (
                <label className="mt-3 flex cursor-pointer items-start gap-2 rounded-xl border border-amber-300/25 bg-amber-300/[0.07] p-3 text-xs leading-5 text-amber-100">
                  <input
                    type="checkbox"
                    checked={allCasesConfirmed}
                    onChange={(event) => setAllCasesConfirmed(event.target.checked)}
                    className="mt-1 h-3.5 w-3.5 rounded border-white/30 bg-transparent text-amber-400 focus:ring-amber-300/40"
                  />
                  <span>我确认授权本次全库跨案模式检索</span>
                </label>
              )}
              <p className={`mt-2 text-xs leading-5 ${searchScope === 'all_cases' ? 'text-amber-200/80' : 'text-white/40'}`}>
                {searchScope === 'same_case' && '候选路径仅来自锚点案件。'}
                {searchScope === 'selected_cases' && '只在列出的案件中寻找相似结构；锚点由上方案件 ID 定位。'}
                {searchScope === 'all_cases' && '将扫描全部案件的相似结构；每条路径仍严格限制在单一案件内。'}
              </p>
            </section>

            <section>
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-sm font-semibold">会话文档</h2>
                <span className="text-xs text-white/45">{documents.length} 个</span>
              </div>
              <input ref={fileInput} type="file" multiple accept=".txt,.pdf,.docx" className="hidden" onChange={(event) => void upload(event.target.files)} />
              <button type="button" onClick={() => fileInput.current?.click()} disabled={!sessionId || uploading} className="flex w-full items-center justify-center gap-2 rounded-2xl border border-dashed border-cyan-200/25 bg-cyan-200/[0.05] px-4 py-4 text-sm text-cyan-50 transition hover:border-cyan-200/45 hover:bg-cyan-200/10 disabled:opacity-50">
                <ArrowUpTrayIcon className="h-5 w-5" />
                {uploading ? '解析与切分中…' : '上传 TXT / PDF / DOCX'}
              </button>
              <div className="mt-3 max-h-48 space-y-2 overflow-auto pr-1">
                {documents.map((document) => (
                  <div key={document.document_id} className="flex items-start gap-3 rounded-xl bg-black/10 p-3">
                    <DocumentTextIcon className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
                    <div className="min-w-0">
                      <p className="truncate text-sm text-white/85" title={document.file_name}>{document.file_name}</p>
                      <p className="mt-1 text-xs text-white/35">{document.chunk_count} 个片段 · {document.status}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <div className="mt-8 grid grid-cols-3 gap-2 text-center text-[11px] text-white/45">
              <div className="rounded-xl bg-white/[0.04] p-2"><CircleStackIcon className="mx-auto mb-1 h-4 w-4" />Neo4j</div>
              <div className="rounded-xl bg-white/[0.04] p-2"><DocumentTextIcon className="mx-auto mb-1 h-4 w-4" />临时文档</div>
              <div className="rounded-xl bg-white/[0.04] p-2"><ShieldCheckIcon className="mx-auto mb-1 h-4 w-4" />证据可追溯</div>
            </div>
          </div>
        </aside>

        <section className="flex min-h-[70vh] min-w-0 flex-1 flex-col">
          <header className="flex items-center justify-between border-b border-slate-200/80 bg-[#f5f3ee]/90 px-5 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90 md:px-9">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700 dark:text-teal-300">Evidence-grounded intelligence</p>
              <h2 className="mt-1 text-lg font-semibold">知识问答与关系研判</h2>
            </div>
            <div className="hidden items-center gap-2 text-xs text-slate-500 md:flex"><BoltIcon className="h-4 w-4 text-amber-500" />自动路由：文档 / 图谱 / 混合</div>
          </header>

          <div className="flex-1 overflow-y-auto px-5 py-8 md:px-9 lg:px-12">
            <div className="mx-auto max-w-4xl space-y-6">
              {messages.length === 0 && (
                <div className="py-8 md:py-16">
                  <div className="mb-7 inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-900 text-amber-300 shadow-xl shadow-teal-900/10"><UserGroupIcon className="h-7 w-7" /></div>
                  <h3 className="max-w-2xl text-3xl font-semibold leading-tight tracking-tight md:text-4xl">从人物、案件与材料中，找到有依据的联系。</h3>
                  <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-400">可直接询问图谱关系，也可先上传调查材料。回答会显示检索路径、证据类型和相关原文。</p>
                  <div className="mt-8 grid gap-3 md:grid-cols-3">
                    {[
                      ['提取多条路径', '谢晚林与刘某之间有哪些路径？'],
                      ['综合材料研判', '材料中的项目与图谱实体有什么关联？'],
                      ['寻找相似模式', '查找谢晚林与刘某之间的相似路径'],
                    ].map(([title, sample]) => (
                      <button key={title} type="button" onClick={() => setQuestion(sample)} className="rounded-2xl border border-slate-200 bg-white/60 p-4 text-left transition hover:-translate-y-0.5 hover:border-teal-700/30 hover:bg-white dark:border-slate-800 dark:bg-slate-900/50">
                        <span className="text-sm font-semibold">{title}</span>
                        <span className="mt-2 block text-xs leading-5 text-slate-500">{sample}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((message) => (
                <article key={message.id} className={message.role === 'user' ? 'ml-auto max-w-2xl' : 'mr-auto max-w-3xl'}>
                  <div className={`rounded-2xl px-5 py-4 shadow-sm ${message.role === 'user' ? 'bg-teal-900 text-white' : 'border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900'}`}>
                    {message.role === 'assistant' && message.result && (
                      <div className="mb-4 flex flex-wrap items-center gap-2 border-b border-slate-100 pb-3 dark:border-slate-800">
                        <span className="rounded-full bg-teal-50 px-2.5 py-1 text-xs font-semibold text-teal-800 dark:bg-teal-950 dark:text-teal-200">{routeLabels[message.result.route]}</span>
                        <span className="text-xs text-slate-400">{message.result.evidence.length} 条证据</span>
                      </div>
                    )}
                    <div className={message.role === 'assistant' ? 'markdown-content' : 'whitespace-pre-wrap text-sm leading-7'}>
                      {message.role === 'assistant' ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown> : message.content}
                    </div>
                    {message.result && message.result.evidence.length > 0 && (
                      <AssistantEvidence evidence={message.result.evidence} />
                    )}
                  </div>
                </article>
              ))}

              {asking && (
                <div className="flex max-w-sm items-center gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-4 text-sm text-slate-500 shadow-sm dark:border-slate-800 dark:bg-slate-900">
                  <span className="flex gap-1"><i className="h-2 w-2 animate-bounce rounded-full bg-teal-700 [animation-delay:-.2s]" /><i className="h-2 w-2 animate-bounce rounded-full bg-teal-700 [animation-delay:-.1s]" /><i className="h-2 w-2 animate-bounce rounded-full bg-teal-700" /></span>
                  正在检索与组织证据…
                </div>
              )}
              <div ref={messagesEnd} />
            </div>
          </div>

          <footer className="border-t border-slate-200/80 bg-[#f5f3ee] px-5 py-4 dark:border-slate-800 dark:bg-slate-950 md:px-9 md:py-6">
            <div className="mx-auto max-w-4xl">
              {notice && <div className="mb-3 flex items-start gap-2 rounded-xl bg-amber-50 px-3.5 py-2.5 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-200"><CheckCircleIcon className="mt-0.5 h-4 w-4 shrink-0" />{notice}</div>}
              <form onSubmit={ask} className="flex items-end gap-3 rounded-2xl border border-slate-300 bg-white p-2 shadow-lg shadow-slate-900/5 focus-within:border-teal-700 dark:border-slate-700 dark:bg-slate-900">
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      event.currentTarget.form?.requestSubmit()
                    }
                  }}
                  rows={1}
                  maxLength={4000}
                  placeholder={online ? '输入问题，Enter 发送，Shift + Enter 换行…' : '请先启动并连接第二阶段 API…'}
                  disabled={!online || asking}
                  className="max-h-40 min-h-[48px] flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-6 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed"
                />
                <button type="submit" disabled={!question.trim() || !sessionId || asking} className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-teal-900 text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-35"><PaperAirplaneIcon className="h-5 w-5" /></button>
              </form>
              <p className="mt-2 text-center text-[11px] text-slate-400">答案由检索证据与模型生成，请结合原始材料复核。</p>
            </div>
          </footer>
        </section>
      </div>
    </main>
  )
}

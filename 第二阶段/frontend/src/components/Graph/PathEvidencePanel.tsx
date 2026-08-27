import {
  ChevronDownIcon,
  DocumentMagnifyingGlassIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import { Fragment, useState } from 'react'

import type { Evidence } from '@/lib/phase2Api'

import { GraphPathView, splitPathEvidence } from './pathEvidence'

const entityStyles: Record<string, string> = {
  PER: 'border-teal-300 bg-teal-50 text-teal-950 dark:border-teal-700 dark:bg-teal-950/60 dark:text-teal-100',
  ORG: 'border-sky-300 bg-sky-50 text-sky-950 dark:border-sky-700 dark:bg-sky-950/60 dark:text-sky-100',
  POSITION: 'border-amber-300 bg-amber-50 text-amber-950 dark:border-amber-700 dark:bg-amber-950/60 dark:text-amber-100',
  MONEY: 'border-rose-300 bg-rose-50 text-rose-950 dark:border-rose-700 dark:bg-rose-950/60 dark:text-rose-100',
}

const entityLabels: Record<string, string> = {
  PER: '人物',
  ORG: '机构',
  POSITION: '职位',
  MONEY: '金额',
  UNKNOWN: '实体',
}

const scopeLabels = {
  same_case: '同案候选',
  selected_cases: '指定案件',
  all_cases: '全库跨案',
}

function EntityNode({ entity }: { entity: GraphPathView['entities'][number] }) {
  return (
    <div className={`w-32 shrink-0 rounded-xl border px-3 py-2.5 shadow-sm ${entityStyles[entity.type] || 'border-slate-300 bg-slate-50 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100'}`}>
      <p className="truncate text-sm font-semibold" title={entity.name}>{entity.name}</p>
      <p className="mt-1 text-[10px] font-semibold uppercase tracking-wider opacity-55">
        {entityLabels[entity.type] || entity.type}
      </p>
    </div>
  )
}

function RelationEdge({
  claimId,
  relation,
  direction,
  active,
  controls,
  onSelect,
}: {
  claimId: string
  relation: string
  direction: 'forward' | 'reverse'
  active: boolean
  controls: string
  onSelect: () => void
}) {
  const directionLabel = direction === 'forward' ? 'HEAD → TAIL' : 'TAIL → HEAD'
  return (
    <div className="flex w-28 shrink-0 flex-col items-center px-1 text-center">
      <button
        type="button"
        onClick={onSelect}
        aria-controls={controls}
        aria-expanded={active}
        aria-label={`${relation}，${directionLabel}，点击查看 Claim ${claimId} 的原始证据`}
        className={`max-w-full truncate rounded-full border px-2 py-1 text-[11px] font-medium transition focus:outline-none focus:ring-2 focus:ring-teal-500/40 ${active ? 'border-teal-500 bg-teal-100 text-teal-900 dark:border-teal-500 dark:bg-teal-950 dark:text-teal-100' : 'border-transparent bg-slate-100 text-slate-700 hover:border-teal-300 hover:bg-teal-50 dark:bg-slate-800 dark:text-slate-200 dark:hover:border-teal-700 dark:hover:bg-teal-950/50'}`}
        title={`${relation} · ${directionLabel} · 点击查看原始证据`}
      >
        {relation}
      </button>
      <div className="mt-1 flex w-full items-center text-teal-700 dark:text-teal-300">
        {direction === 'reverse' && <span className="text-base leading-none">←</span>}
        <span className="h-px flex-1 bg-current" />
        {direction === 'forward' && <span className="text-base leading-none">→</span>}
      </div>
      <span className="mt-0.5 text-[9px] font-medium tracking-wide text-slate-400">{directionLabel}</span>
    </div>
  )
}

function SimilarityBreakdown({ path }: { path: GraphPathView }) {
  if (!path.similarity) return null
  const metrics = [
    ['关系序列', path.similarity.relationSequenceScore],
    ['实体类型', path.similarity.entityTypeSequenceScore],
    ['关系重合', path.similarity.relationOverlapScore],
    ['长度接近', path.similarity.lengthScore],
  ] as const
  return (
    <div className="mt-4 rounded-xl bg-violet-50/70 p-3 dark:bg-violet-950/25">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold text-violet-800 dark:text-violet-200">综合相似度 {Math.round(path.similarity.score * 100)}%</span>
        <span className="rounded-full bg-white/80 px-2 py-0.5 text-[10px] text-violet-600 dark:bg-slate-900/70 dark:text-violet-300">
          {path.similarity.orientation === 'reversed' ? '反向匹配' : '同向匹配'}
        </span>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {metrics.map(([label, value]) => (
          <div key={label}>
            <div className="mb-1 flex justify-between text-[10px] text-slate-500 dark:text-slate-400">
              <span>{label}</span><span>{Math.round(value * 100)}%</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-white dark:bg-slate-800">
              <div className="h-full rounded-full bg-violet-500" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function PathCard({ path, number }: { path: GraphPathView; number: number }) {
  const [selectedClaimIndex, setSelectedClaimIndex] = useState<number | null>(null)
  const similar = path.kind === 'similar_path'
  const caseLabel = path.candidateCaseId || path.caseIds[0] || path.source || '未标注案件'
  const selectedClaim = selectedClaimIndex === null ? null : path.claims[selectedClaimIndex]
  const claimPanelId = `claim-evidence-${path.evidenceId.replace(/[^A-Za-z0-9_-]/g, '-')}`
  const safeSourceUrl = selectedClaim?.sourceUrl && /^https?:\/\//i.test(selectedClaim.sourceUrl)
    ? selectedClaim.sourceUrl
    : null
  return (
    <article className={`rounded-2xl border p-4 ${similar ? 'border-violet-200 bg-violet-50/30 dark:border-violet-900 dark:bg-violet-950/10' : 'border-teal-200 bg-teal-50/25 dark:border-teal-900 dark:bg-teal-950/10'}`}>
      <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {similar && <SparklesIcon className="h-4 w-4 text-violet-500" />}
          <span className="text-sm font-semibold">{similar ? '相似路径' : '关系路径'} {number}</span>
          <span className="rounded-full bg-white px-2 py-0.5 text-[10px] text-slate-500 shadow-sm dark:bg-slate-900 dark:text-slate-400">{path.hopCount} 跳</span>
          {path.searchScope && <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[10px] text-violet-700 dark:bg-violet-950 dark:text-violet-200">{scopeLabels[path.searchScope]}</span>}
        </div>
        <div className="text-right">
          {path.score !== null && <p className="font-mono text-sm font-semibold text-slate-700 dark:text-slate-200">{Math.round(path.score * 100)}%</p>}
          <p className="max-w-56 truncate text-[10px] text-slate-400" title={caseLabel}>{caseLabel}</p>
        </div>
      </header>

      <div className="overflow-x-auto pb-2">
        <div className="flex min-w-max items-center py-1">
          {path.entities.map((entity, index) => (
            <Fragment key={`${entity.id}-${index}`}>
              <EntityNode entity={entity} />
              {index < path.claims.length && (
                <RelationEdge
                  claimId={path.claims[index].id}
                  relation={path.claims[index].relationType}
                  direction={path.directions[index]}
                  active={selectedClaimIndex === index}
                  controls={claimPanelId}
                  onSelect={() => setSelectedClaimIndex((current) => current === index ? null : index)}
                />
              )}
            </Fragment>
          ))}
        </div>
      </div>

      {selectedClaim && (
        <section id={claimPanelId} aria-live="polite" className="mt-3 rounded-xl border border-teal-200 bg-white p-3 text-xs shadow-sm dark:border-teal-800 dark:bg-slate-900">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold text-teal-900 dark:text-teal-100">已选 Claim · {selectedClaim.relationType}</span>
              <span className="font-mono text-[10px] text-slate-400">{selectedClaim.id}</span>
              {selectedClaim.documentId && <span className="font-mono text-[10px] text-slate-400">{selectedClaim.documentId}</span>}
            </div>
            <button type="button" onClick={() => setSelectedClaimIndex(null)} className="text-[10px] text-slate-400 hover:text-slate-700 dark:hover:text-slate-200">关闭</button>
          </div>
          <p className="mt-2 whitespace-pre-wrap leading-5 text-slate-600 dark:text-slate-300">{selectedClaim.evidenceText || '该 Claim 未返回证据原文。'}</p>
          {safeSourceUrl && <a href={safeSourceUrl} target="_blank" rel="noreferrer" className="mt-2 inline-flex text-teal-700 underline decoration-teal-300 underline-offset-2 hover:text-teal-900 dark:text-teal-300 dark:hover:text-teal-100">打开原始来源</a>}
        </section>
      )}

      <SimilarityBreakdown path={path} />

      <details className="group mt-3 rounded-xl border border-slate-200 bg-white/70 dark:border-slate-700 dark:bg-slate-900/60">
        <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-xs font-medium text-slate-600 dark:text-slate-300">
          <span className="flex items-center gap-2"><DocumentMagnifyingGlassIcon className="h-4 w-4" />查看 Claim 与原始证据</span>
          <ChevronDownIcon className="h-4 w-4 transition group-open:rotate-180" />
        </summary>
        <div className="space-y-2 border-t border-slate-100 p-3 dark:border-slate-800">
          {path.claims.map((claim, index) => (
            <div key={claim.id} className="rounded-lg bg-slate-50 p-3 text-xs dark:bg-slate-800/70">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold text-slate-800 dark:text-slate-100">{index + 1}. {claim.relationType}</span>
                {claim.status && <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] text-emerald-700 dark:bg-emerald-950 dark:text-emerald-200">{claim.status}</span>}
                {claim.documentId && <span className="font-mono text-[10px] text-slate-400">{claim.documentId}</span>}
              </div>
              <p className="mt-2 whitespace-pre-wrap leading-5 text-slate-600 dark:text-slate-300">{claim.evidenceText || '该 Claim 未返回证据原文。'}</p>
            </div>
          ))}
        </div>
      </details>
    </article>
  )
}

export default function PathEvidencePanel({ evidence }: { evidence: Evidence[] }) {
  const { paths } = splitPathEvidence(evidence)
  if (paths.length === 0) return null
  const directPaths = paths.filter((path) => path.kind === 'path')
  const similarPaths = paths.filter((path) => path.kind === 'similar_path')
  return (
    <section className="mt-5 space-y-4 border-t border-slate-100 pt-4 dark:border-slate-800">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">关系路径视图</p>
        <p className="text-xs text-slate-400">{directPaths.length} 条锚点 · {similarPaths.length} 条相似</p>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl bg-slate-50 px-3 py-2 text-[10px] text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
        <span className="font-semibold">实体类型</span>
        <span className="inline-flex items-center gap-1"><i className="h-2.5 w-2.5 rounded bg-teal-200 ring-1 ring-teal-400" />人物</span>
        <span className="inline-flex items-center gap-1"><i className="h-2.5 w-2.5 rounded bg-sky-200 ring-1 ring-sky-400" />机构</span>
        <span className="inline-flex items-center gap-1"><i className="h-2.5 w-2.5 rounded bg-amber-200 ring-1 ring-amber-400" />职位</span>
        <span className="inline-flex items-center gap-1"><i className="h-2.5 w-2.5 rounded bg-rose-200 ring-1 ring-rose-400" />金额</span>
        <span className="ml-auto">→ HEAD → TAIL · ← TAIL → HEAD · 点击关系标签查看 Claim</span>
      </div>
      {directPaths.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 pt-1 text-xs font-semibold text-teal-700 dark:text-teal-300">普通路径</div>
          {directPaths.map((path, index) => <PathCard key={path.evidenceId} path={path} number={index + 1} />)}
        </div>
      )}
      {similarPaths.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 pt-1 text-xs font-semibold text-violet-700 dark:text-violet-300"><SparklesIcon className="h-4 w-4" />结构相似候选</div>
          {similarPaths.map((path, index) => <PathCard key={path.evidenceId} path={path} number={index + 1} />)}
        </div>
      )}
    </section>
  )
}

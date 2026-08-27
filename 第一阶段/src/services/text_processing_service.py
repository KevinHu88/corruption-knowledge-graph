"""原始案件文档到可追溯模型文本块的统一处理服务。

本模块仅负责解析、确定性清洗、字符映射、文本切分、相关性判断和
模型输入块构造；不执行实体识别、关系抽取、标注持久化或模型训练。
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import inspect
import logging
import mimetypes
import re
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from xml.etree import ElementTree

from pydantic import BaseModel, Field

from config import BASE_DIR, ProjectConfig, load_project_config
from models import (
    CleanTextResult,
    DeterministicFilterResult,
    ModelInputChunk,
    ParseResult,
    ProcessedCase,
    RawDocument,
    RelevantSpan,
    RelevanceJudgment,
    RuleRelevanceResult,
    TextSegment,
    TextWindow,
)
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class DocumentParseError(RuntimeError):
    """文档存在但无法解析。"""


class UnsupportedDocumentError(DocumentParseError):
    """当前环境或配置不支持该文档类型。"""


class EmptyDocumentError(DocumentParseError):
    """解析或清洗后没有可处理正文。"""


class OffsetMappingError(ValueError):
    """清洗文本与原文字符映射不一致。"""


class InvalidTextWindowError(ValueError):
    """相关性窗口不满足全文字符切片约束。"""


class RelevanceServiceError(RuntimeError):
    """规则或 LLM 相关性判断失败。"""


class RelevanceDecisionPayload(BaseModel):
    """兼容代理使用的扁平相关性 Structured Outputs 传输对象。"""

    relevant: bool
    score: float = Field(ge=0, le=1)
    reason: str


class ModelChunkingError(RuntimeError):
    """无法构造满足模型输入限制的文本块。"""


# 中文注释：文本处理算法的强类型配置，集中约束窗口、阈值、token 限制和并发参数。
class TextProcessingConfig(BaseModel):
    """从 workflow/training 配置归一化出的文本处理参数。"""

    allowed_content_types: set[str] = Field(default_factory=set)
    minimum_text_length: int = Field(default=20, ge=1)
    relevance_target_chars: int = Field(default=800, ge=1)
    relevance_max_chars: int = Field(default=1200, ge=1)
    relevance_overlap_chars: int = Field(default=150, ge=0)
    accept_threshold: float = Field(default=0.7, ge=0, le=1)
    reject_threshold: float = Field(default=0.2, ge=0, le=1)
    llm_review_between_thresholds: bool = True
    positive_terms: dict[str, float] = Field(default_factory=dict)
    negative_terms: dict[str, float] = Field(default_factory=dict)
    irrelevant_title_terms: list[str] = Field(default_factory=list)
    irrelevant_page_types: list[str] = Field(default_factory=list)
    span_max_gap_chars: int = Field(default=80, ge=0)
    evidence_before_sentences: int = Field(default=1, ge=0)
    evidence_after_sentences: int = Field(default=1, ge=0)
    model_max_tokens: int = Field(default=480, ge=8)
    model_overlap_tokens: int = Field(default=64, ge=0)
    model_fallback_max_chars: int = Field(default=800, ge=16)
    respect_sentence_boundary: bool = True
    tokenizer_from_ner_model: bool = True
    context_sentences: int = Field(default=1, ge=0)
    max_concurrency: int = Field(default=4, ge=1)
    mark_partial_on_llm_failure: bool = True

    @classmethod
    def from_project(cls, project: ProjectConfig) -> "TextProcessingConfig":
        """从项目现有 YAML 结构构造配置，不在 Service 中读取 YAML。"""

        root = project.workflow.get("text_processing", {})
        relevance_window = root.get("relevance_window", {})
        relevance = root.get("relevance", {})
        deterministic = root.get("deterministic_filter", {})
        span_merge = root.get("span_merge", {})
        expansion = root.get("evidence_expansion", {})
        model_input = root.get("model_input", {})
        batch = root.get("batch", {})
        llm_failure = root.get("llm_failure", {})
        legacy = project.workflow.get("relevance_filter", {})
        ner_max = int(
            project.training.get("modeling", {})
            .get("ner", {})
            .get("max_length", 512)
        )
        configured_tokens = int(model_input.get("max_tokens", 480))
        # 为 [CLS]/[SEP] 留空间，并避免超过实际 NER 配置。
        safe_tokens = min(configured_tokens, max(8, ner_max - 2))
        return cls(
            allowed_content_types=set(root.get("allowed_content_types", [])),
            minimum_text_length=int(root.get("minimum_text_length", 20)),
            relevance_target_chars=int(
                relevance_window.get(
                    "target_chars", legacy.get("window_size", 800)
                )
            ),
            relevance_max_chars=int(
                relevance_window.get("max_chars", 1200)
            ),
            relevance_overlap_chars=int(
                relevance_window.get(
                    "overlap_chars", legacy.get("window_overlap", 150)
                )
            ),
            accept_threshold=float(
                relevance.get("accept_threshold", 0.7)
            ),
            reject_threshold=float(
                relevance.get("reject_threshold", 0.2)
            ),
            llm_review_between_thresholds=bool(
                relevance.get("llm_review_between_thresholds", True)
            ),
            positive_terms={
                str(term): float(weight)
                for term, weight in relevance.get(
                    "positive_terms", {}
                ).items()
            },
            negative_terms={
                str(term): float(weight)
                for term, weight in relevance.get(
                    "negative_terms", {}
                ).items()
            },
            irrelevant_title_terms=list(
                deterministic.get("irrelevant_title_terms", [])
            ),
            irrelevant_page_types=list(
                deterministic.get("irrelevant_page_types", [])
            ),
            span_max_gap_chars=int(
                span_merge.get("max_gap_chars", 80)
            ),
            evidence_before_sentences=int(
                expansion.get("before_sentences", 1)
            ),
            evidence_after_sentences=int(
                expansion.get("after_sentences", 1)
            ),
            model_max_tokens=safe_tokens,
            model_overlap_tokens=min(
                int(model_input.get("overlap_tokens", 64)),
                safe_tokens - 1,
            ),
            model_fallback_max_chars=int(
                model_input.get("fallback_max_chars", 800)
            ),
            respect_sentence_boundary=bool(
                model_input.get("respect_sentence_boundary", True)
            ),
            tokenizer_from_ner_model=bool(
                model_input.get("tokenizer_from_ner_model", True)
            ),
            context_sentences=int(
                model_input.get("context_sentences", 1)
            ),
            max_concurrency=int(batch.get("max_concurrency", 4)),
            mark_partial_on_llm_failure=bool(
                llm_failure.get("mark_partial", True)
            ),
        )


class _BodyHTMLParser(HTMLParser):
    """仅依赖标准库的可见 HTML 正文提取器。"""

    _block_tags = {
        "article", "blockquote", "br", "div", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "p", "section", "table", "tr",
    }
    _ignored_tags = {
        "script", "style", "noscript", "svg", "nav", "footer",
        "form", "aside",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        name = tag.lower()
        if name in self._ignored_tags:
            self._ignored_depth += 1
        if name == "title":
            self._in_title = True
        if name in self._block_tags and not self._ignored_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name == "title":
            self._in_title = False
        if name in self._block_tags and not self._ignored_depth:
            self.parts.append("\n")
        if name in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if not self._ignored_depth and not self._in_title:
            self.parts.append(data)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


# 中文注释：原始文档进入模型前的核心处理服务，负责解析、清洗、相关性判断和分块。
class TextProcessingService:
    """编排文档解析、相关性筛选和模型文本块构造。"""

    parser_version = "1.0"

    def __init__(
        self,
        config: TextProcessingConfig | None = None,
        llm_service: LLMService | None = None,
        tokenizer: Any = None,
        *,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.project_config = project_config or load_project_config()
        self.config = config or TextProcessingConfig.from_project(
            self.project_config
        )
        self.llm_service = llm_service
        self.tokenizer = tokenizer
        self._tokenizer_attempted = tokenizer is not None
        self._tokenizer_warning: str | None = None
        self._prompt_template: str | None = None
        self._validate_config()

    # 中文注释：单文档处理总入口，依次执行解析、清洗、过滤、相关跨度合并和模型切块。
    async def process_document(
        self,
        document: RawDocument,
    ) -> ProcessedCase:
        """将单个原始文档处理成可追溯的 ProcessedCase。"""

        try:
            parsed = self.parse_document(document)
            if not parsed.text.strip():
                raise EmptyDocumentError(
                    f"文档解析结果为空：doc_id={document.doc_id}"
                )
            cleaned = self.clean_text(parsed.text)
            if not cleaned.text.strip():
                raise EmptyDocumentError(
                    f"文档清洗结果为空：doc_id={document.doc_id}"
                )
            paragraphs = self.split_paragraphs(cleaned.text)
            sentences = self.split_sentences(cleaned.text, paragraphs)
            deterministic = self.deterministic_filter(
                document, cleaned.text
            )
            case_id = _stable_id(
                "case", document.doc_id, document.doc_version_id or ""
            )
            errors = [*parsed.warnings, *cleaned.warnings]
            if deterministic.status == "irrelevant":
                return self._processed_case(
                    document,
                    parsed,
                    cleaned,
                    case_id,
                    paragraphs,
                    sentences,
                    [],
                    [],
                    [],
                    "irrelevant",
                    errors + deterministic.reason_codes,
                )

            windows = self.build_relevance_windows(
                cleaned.text, sentences
            )
            windows, relevance_errors = await self._filter_windows(
                document, windows
            )
            errors.extend(relevance_errors)
            spans = self.merge_relevant_spans(windows, cleaned.text)
            chunks = self.build_model_input_chunks(
                case_id,
                cleaned.text,
                spans,
                sentences,
            )
            if self._tokenizer_warning:
                errors.append(self._tokenizer_warning)
            uncertain_remains = any(
                item.relevance_status == "uncertain" for item in windows
            )
            status: Literal["ready", "irrelevant", "partial", "failed"]
            if errors or uncertain_remains:
                status = "partial"
            elif spans:
                status = "ready"
            else:
                status = "irrelevant"
            return self._processed_case(
                document,
                parsed,
                cleaned,
                case_id,
                paragraphs,
                sentences,
                windows,
                spans,
                chunks,
                status,
                errors,
            )
        except (
            DocumentParseError,
            OffsetMappingError,
            InvalidTextWindowError,
            ModelChunkingError,
        ):
            logger.exception(
                "文本处理失败 doc_id=%s doc_version_id=%s",
                document.doc_id,
                document.doc_version_id,
            )
            raise

    # 中文注释：批量处理入口，使用 Semaphore 限制并发，并把单文档异常转换为失败结果。
    async def process_documents(
        self,
        documents: list[RawDocument],
    ) -> list[ProcessedCase]:
        """受控并发处理文档；单个失败不会取消其他文档。"""

        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def run(document: RawDocument) -> ProcessedCase:
            async with semaphore:
                try:
                    return await self.process_document(document)
                except Exception as exc:
                    logger.error(
                        "批量文本处理失败 doc_id=%s type=%s",
                        document.doc_id,
                        type(exc).__name__,
                    )
                    return self._failed_case(document, exc)

        return list(await asyncio.gather(*(run(item) for item in documents)))

    # 中文注释：按内容类型选择解析器，将文本、HTML、DOCX 或 PDF 统一转换为原始字符串。
    def parse_document(self, document: RawDocument) -> ParseResult:
        """根据 content_type、扩展名和内容特征选择解析器。"""

        content_type = self._detect_content_type(document)
        if (
            self.config.allowed_content_types
            and content_type not in self.config.allowed_content_types
        ):
            raise UnsupportedDocumentError(
                f"配置不允许文档类型：{content_type}"
            )
        try:
            raw = (
                document.raw_text
                if document.raw_text is not None
                else self._read_file(document.local_path, content_type)
            )
            if content_type == "text/html":
                parser = _BodyHTMLParser()
                parser.feed(raw)
                return ParseResult(
                    text=html.unescape("".join(parser.parts)),
                    title=document.title
                    or "".join(parser.title_parts).strip()
                    or None,
                    published_at=document.published_at,
                    metadata={**document.metadata, "content_type": content_type},
                    parser_name="stdlib_html",
                    parser_version=self.parser_version,
                )
            if content_type in {
                "text/plain",
                "text/markdown",
            }:
                return ParseResult(
                    text=raw,
                    title=document.title,
                    published_at=document.published_at,
                    metadata={**document.metadata, "content_type": content_type},
                    parser_name=(
                        "plain_text"
                        if content_type == "text/plain"
                        else "markdown_text"
                    ),
                    parser_version=self.parser_version,
                )
            if content_type == (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ):
                if not document.local_path:
                    raise DocumentParseError(
                        "DOCX 解析需要 local_path，不能使用 raw_text"
                    )
                return self._parse_docx(
                    Path(document.local_path), document
                )
            if content_type == "application/pdf":
                if not document.local_path:
                    raise DocumentParseError(
                        "PDF 解析需要 local_path，不能使用 raw_text"
                    )
                return self._parse_pdf(Path(document.local_path), document)
        except UnsupportedDocumentError:
            raise
        except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
            raise DocumentParseError(
                f"文档解析失败：doc_id={document.doc_id}"
            ) from exc
        raise UnsupportedDocumentError(f"不支持的文档类型：{content_type}")

    # 中文注释：清理换行、零宽字符和无意义空白，同时保留字符位置映射以支持证据追溯。
    def clean_text(self, original_text: str) -> CleanTextResult:
        """确定性清洗文本，并生成 cleaned→original 和反向映射。"""

        if not isinstance(original_text, str):
            raise TypeError("original_text 必须是字符串")
        chars: list[str] = []
        origins: list[list[int]] = []
        index = 0
        while index < len(original_text):
            character = original_text[index]
            if character == "\r":
                source = [index]
                if (
                    index + 1 < len(original_text)
                    and original_text[index + 1] == "\n"
                ):
                    source.append(index + 1)
                    index += 1
                chars.append("\n")
                origins.append(source)
            elif character in {"\u200b", "\u200c", "\u200d", "\ufeff"}:
                pass
            elif ord(character) < 32 and character not in {"\n", "\t"}:
                pass
            elif character in {" ", "\t", "\u00a0", "\u3000"}:
                if chars and chars[-1] == " ":
                    origins[-1].append(index)
                else:
                    chars.append(" ")
                    origins.append([index])
            else:
                chars.append(character)
                origins.append([index])
            index += 1

        chars, origins = self._remove_decorative_lines(chars, origins)
        while chars and chars[0].isspace():
            chars.pop(0)
            origins.pop(0)
        while chars and chars[-1].isspace():
            chars.pop()
            origins.pop()
        cleaned = "".join(chars)
        cleaned_to_original = [
            source[0] for source in origins
        ]
        original_to_cleaned = [-1] * len(original_text)
        for clean_index, source_indexes in enumerate(origins):
            for original_index in source_indexes:
                original_to_cleaned[original_index] = clean_index
        if len(cleaned_to_original) != len(cleaned):
            raise OffsetMappingError("cleaned_to_original 长度不一致")
        return CleanTextResult(
            text=cleaned,
            original_to_cleaned_mapping=original_to_cleaned,
            cleaned_to_original_mapping=cleaned_to_original,
        )

    def split_paragraphs(self, text: str) -> list[TextSegment]:
        """按空行顺序扫描段落并保存全文位置。"""

        paragraphs: list[TextSegment] = []
        cursor = 0
        for separator in re.finditer(r"\n[ \t]*\n+", text):
            self._append_paragraph(
                paragraphs, text, cursor, separator.start()
            )
            cursor = separator.end()
        self._append_paragraph(paragraphs, text, cursor, len(text))
        return paragraphs

    # 中文注释：在段落范围内按中英文标点切句，并特殊处理缩写和小数点等边界。
    def split_sentences(
        self,
        text: str,
        paragraphs: list[TextSegment],
    ) -> list[TextSegment]:
        """单次顺序扫描中英文句末标点，不使用 text.find 定位。"""

        sentences: list[TextSegment] = []
        closing = set("”’」』】）》)]")
        for paragraph in paragraphs:
            start = paragraph.start
            cursor = paragraph.start
            while cursor < paragraph.end:
                character = text[cursor]
                boundary = character in "。！？；!?"
                if character == ".":
                    previous = text[cursor - 1] if cursor > start else ""
                    following = (
                        text[cursor + 1]
                        if cursor + 1 < paragraph.end
                        else ""
                    )
                    word_start = cursor
                    while (
                        word_start > start
                        and text[word_start - 1].isalpha()
                        and text[word_start - 1].isascii()
                    ):
                        word_start -= 1
                    abbreviation = text[word_start:cursor].lower()
                    common_abbreviations = {
                        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
                        "st", "vs", "etc", "e.g", "i.e", "no",
                    }
                    boundary = (
                        not (previous.isdigit() and following.isdigit())
                        and abbreviation not in common_abbreviations
                        and len(abbreviation) != 1
                        and (
                            not following
                            or following.isspace()
                            or following in closing
                        )
                    )
                if character == "\n":
                    boundary = True
                if boundary:
                    end = cursor + 1
                    while end < paragraph.end and text[end] in closing:
                        end += 1
                    self._append_sentence(
                        sentences, text, start, end, paragraph
                    )
                    start = end
                    while start < paragraph.end and text[start].isspace():
                        start += 1
                    cursor = start
                    continue
                cursor += 1
            self._append_sentence(
                sentences, text, start, paragraph.end, paragraph
            )
        return sentences

    # 中文注释：使用文本长度、标题词、来源和元数据进行首轮确定性过滤，减少不必要的 LLM 调用。
    def deterministic_filter(
        self,
        document: RawDocument,
        parsed_text: str,
    ) -> DeterministicFilterResult:
        """根据来源、标题、正文长度和页面类型进行确定性过滤。"""

        reasons: list[str] = []
        if len(parsed_text.strip()) < self.config.minimum_text_length:
            return DeterministicFilterResult(
                status="irrelevant", reason_codes=["text_too_short"]
            )
        title = document.title or ""
        matched_titles = [
            term
            for term in self.config.irrelevant_title_terms
            if term and term in title
        ]
        if matched_titles:
            return DeterministicFilterResult(
                status="irrelevant",
                reason_codes=[
                    f"irrelevant_title:{term}" for term in matched_titles
                ],
            )
        page_type = str(document.metadata.get("page_type", ""))
        if page_type in self.config.irrelevant_page_types:
            return DeterministicFilterResult(
                status="irrelevant",
                reason_codes=[f"irrelevant_page_type:{page_type}"],
            )
        configured_sources = self.project_config.sources.get("sources", [])
        source = next(
            (
                item
                for item in configured_sources
                if item.get("source_id") == document.source_id
            ),
            None,
        )
        if source:
            reasons.append("known_source")
            domain = (urlsplit(document.source_url or "").hostname or "")
            allowed_domains = source.get("domains", [])
            if allowed_domains and domain and not any(
                domain == item or domain.endswith(f".{item}")
                for item in allowed_domains
            ):
                reasons.append("source_domain_mismatch")
        explicit = str(document.metadata.get("relevance_status", "")).lower()
        if explicit in {"relevant", "irrelevant"}:
            return DeterministicFilterResult(
                status=explicit, reason_codes=["metadata_relevance_status"]
            )
        return DeterministicFilterResult(
            status="uncertain", reason_codes=reasons or ["rules_required"]
        )

    # 中文注释：按句子边界构建带重叠的字符窗口，供规则评分和 LLM 相关性复核使用。
    def build_relevance_windows(
        self,
        cleaned_text: str,
        sentences: list[TextSegment],
    ) -> list[TextWindow]:
        """按句子边界构造相关性窗口，过长句子才按字符降级。"""

        pieces = self._window_pieces(sentences)
        windows: list[TextWindow] = []
        index = 0
        while index < len(pieces):
            start = pieces[index][0]
            end = pieces[index][1]
            next_index = index + 1
            while next_index < len(pieces):
                candidate_end = pieces[next_index][1]
                if candidate_end - start > self.config.relevance_max_chars:
                    break
                end = candidate_end
                next_index += 1
                if end - start >= self.config.relevance_target_chars:
                    break
            segment_ids = [
                segment.segment_id
                for segment in sentences
                if segment.start < end and segment.end > start
            ]
            windows.append(
                TextWindow(
                    window_id=_stable_id("window", start, end, cleaned_text[start:end]),
                    text=cleaned_text[start:end],
                    start=start,
                    end=end,
                    segment_ids=segment_ids,
                )
            )
            if next_index >= len(pieces):
                break
            threshold = end - self.config.relevance_overlap_chars
            overlap_index = next_index
            while (
                overlap_index > index + 1
                and pieces[overlap_index - 1][0] >= threshold
            ):
                overlap_index -= 1
            index = overlap_index if overlap_index > index else index + 1
        self._validate_windows(windows, cleaned_text)
        return windows

    # 中文注释：根据正负关键词计算可解释的规则分数，并将结果限制在 0 到 1。
    def score_window_relevance(
        self,
        window: TextWindow,
    ) -> RuleRelevanceResult:
        """按 workflow 中配置的词项和阈值计算窗口相关性。"""

        positives = [
            term for term in self.config.positive_terms if term in window.text
        ]
        negatives = [
            term for term in self.config.negative_terms if term in window.text
        ]
        positive_score = sum(
            self.config.positive_terms[term] for term in positives
        )
        negative_score = sum(
            self.config.negative_terms[term] for term in negatives
        )
        score = max(0.0, min(1.0, positive_score - negative_score))
        if score >= self.config.accept_threshold:
            status = "relevant"
        elif score <= self.config.reject_threshold:
            status = "irrelevant"
        else:
            status = "uncertain"
        reasons = [f"positive:{term}" for term in positives]
        reasons.extend(f"negative:{term}" for term in negatives)
        if not reasons:
            reasons.append("no_configured_terms")
        return RuleRelevanceResult(
            score=score,
            matched_positive_terms=positives,
            matched_negative_terms=negatives,
            status=status,
            reason_codes=reasons,
        )

    async def filter_relevant_windows(
        self,
        document: RawDocument,
        windows: list[TextWindow],
    ) -> list[TextWindow]:
        """规则先行，仅将模糊窗口交给可选 LLMService。"""

        filtered, _ = await self._filter_windows(document, windows)
        return filtered

    # 中文注释：只把规则无法确定的窗口交给 LLM，并要求返回 RelevanceJudgment 结构。
    async def review_uncertain_window(
        self,
        window: TextWindow,
        document: RawDocument,
    ) -> RelevanceJudgment:
        """使用现有结构化 LLM 接口判断单个模糊窗口。"""

        if self.llm_service is None:
            raise RelevanceServiceError("未配置 LLMService")
        prompt = self._render_relevance_prompt(document, window.text)

        def invoke() -> RelevanceJudgment:
            payload = self.llm_service.generate_structured_response(
                "你是中文裁判文书相关性判断器。仅返回结构化结果。",
                prompt,
                RelevanceDecisionPayload,
                max_tokens=min(
                    512,
                    self.project_config.environment.llm_max_tokens,
                ),
            )
            return RelevanceJudgment(
                relevant=payload.relevant,
                score=payload.score,
                relevance_types=[],
                evidence_spans=[],
                reason=payload.reason,
            )

        try:
            result = await asyncio.to_thread(invoke)
            if inspect.isawaitable(result):
                result = await result
            return RelevanceJudgment.model_validate(result)
        except Exception as exc:
            raise RelevanceServiceError(
                f"LLM 相关性判断失败：window_id={window.window_id}"
            ) from exc

    # 中文注释：合并重叠或间距很小的相关区间，降低后续切块和推理的重复量。
    def merge_relevant_spans(
        self,
        windows: list[TextWindow],
        cleaned_text: str,
    ) -> list[RelevantSpan]:
        """合并重叠或间距较小的连续证据区间。"""

        candidates: list[dict[str, Any]] = []
        for window in windows:
            if window.relevance_status != "relevant":
                continue
            start = (
                window.evidence_start
                if window.evidence_start is not None
                else window.start
            )
            end = (
                window.evidence_end
                if window.evidence_end is not None
                else window.end
            )
            if start < window.start or end > window.end or end <= start:
                start, end = window.start, window.end
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "window_ids": {window.window_id},
                    "types": {
                        window.relevance_type or "case_relevance"
                    },
                    "scores": [
                        window.llm_score
                        if window.llm_score is not None
                        else window.rule_score
                    ],
                }
            )
        candidates.sort(key=lambda item: (item["start"], item["end"]))
        merged: list[dict[str, Any]] = []
        for candidate in candidates:
            if (
                merged
                and candidate["start"]
                <= merged[-1]["end"] + self.config.span_max_gap_chars
            ):
                previous = merged[-1]
                previous["end"] = max(previous["end"], candidate["end"])
                previous["window_ids"].update(candidate["window_ids"])
                previous["types"].update(candidate["types"])
                previous["scores"].extend(candidate["scores"])
            else:
                merged.append(candidate)
        return [
            RelevantSpan(
                span_id=_stable_id(
                    "span", item["start"], item["end"],
                    cleaned_text[item["start"]:item["end"]],
                ),
                text=cleaned_text[item["start"]:item["end"]],
                start=item["start"],
                end=item["end"],
                source_window_ids=sorted(item["window_ids"]),
                relevance_types=sorted(item["types"]),
                score=max(item["scores"]) if item["scores"] else 0.0,
            )
            for item in merged
        ]

    # 中文注释：将相关区间扩展上下文后切成模型块，优先使用 token 窗口，失败时退回字符窗口。
    def build_model_input_chunks(
        self,
        case_id: str,
        cleaned_text: str,
        relevant_spans: list[RelevantSpan],
        sentences: list[TextSegment],
    ) -> list[ModelInputChunk]:
        """扩展证据上下文，并按 tokenizer token 上限构造模型文本块。"""

        if not relevant_spans:
            return []
        tokenizer = self._ensure_tokenizer()
        ranges = self._expanded_ranges(relevant_spans, sentences)
        chunks: list[ModelInputChunk] = []
        for range_start, range_end in ranges:
            range_sentences = [
                item
                for item in sentences
                if item.start < range_end and item.end > range_start
            ]
            if tokenizer is None:
                raw_ranges = self._fallback_ranges(range_start, range_end)
            else:
                raw_ranges = self._token_ranges(
                    cleaned_text,
                    range_start,
                    range_end,
                    range_sentences,
                )
            for start, end, token_count in raw_ranges:
                if end <= start or not cleaned_text[start:end].strip():
                    continue
                span_ids = [
                    item.span_id
                    for item in relevant_spans
                    if item.start < end and item.end > start
                ]
                segment_ids = [
                    item.segment_id
                    for item in sentences
                    if item.start < end and item.end > start
                ]
                text = cleaned_text[start:end]
                chunks.append(
                    ModelInputChunk(
                        chunk_id=_stable_id(
                            "chunk",
                            case_id,
                            start,
                            end,
                            self.config.model_max_tokens,
                            hashlib.sha256(text.encode()).hexdigest(),
                        ),
                        case_id=case_id,
                        text_id=_stable_id(
                            "text", case_id, start, end,
                            hashlib.sha256(text.encode()).hexdigest(),
                        ),
                        text=text,
                        start=start,
                        end=end,
                        source_span_ids=span_ids,
                        source_segment_ids=segment_ids,
                        token_count=token_count,
                        model_ready=tokenizer is not None,
                    )
                )
        chunks = self._deduplicate_chunks(chunks)
        for index, chunk in enumerate(chunks):
            left = (
                max(0, chunks[index - 1].end - chunk.start)
                if index
                else 0
            )
            right = (
                max(0, chunk.end - chunks[index + 1].start)
                if index + 1 < len(chunks)
                else 0
            )
            chunks[index] = chunk.model_copy(
                update={"overlap_left": left, "overlap_right": right}
            )
        return chunks

    async def _filter_windows(
        self,
        document: RawDocument,
        windows: list[TextWindow],
    ) -> tuple[list[TextWindow], list[str]]:
        output: list[TextWindow] = []
        errors: list[str] = []
        for window in windows:
            rule = self.score_window_relevance(window)
            relevance_type = (
                rule.matched_positive_terms[0]
                if rule.matched_positive_terms
                else None
            )
            updated = window.model_copy(
                update={
                    "rule_score": rule.score,
                    "relevance_status": rule.status,
                    "relevance_type": relevance_type,
                    "processing_source": "rule",
                }
            )
            needs_llm = (
                rule.status == "uncertain"
                and self.config.llm_review_between_thresholds
                and self.llm_service is not None
            )
            if needs_llm:
                try:
                    judgment = await self.review_uncertain_window(
                        updated, document
                    )
                    updated, warning = self._apply_llm_judgment(
                        updated, judgment
                    )
                    if warning:
                        errors.append(warning)
                except RelevanceServiceError as exc:
                    errors.append(str(exc))
            elif (
                rule.status == "uncertain"
                and self.config.llm_review_between_thresholds
                and self.llm_service is None
            ):
                errors.append(
                    f"模糊窗口未复核（未配置 LLMService）：{window.window_id}"
                )
            output.append(updated)
        return output, errors

    @staticmethod
    def _apply_llm_judgment(
        window: TextWindow,
        judgment: RelevanceJudgment,
    ) -> tuple[TextWindow, str | None]:
        evidence_start: int | None = None
        evidence_end: int | None = None
        warning: str | None = None
        valid = []
        for evidence in judgment.evidence_spans:
            if (
                0 <= evidence.start < evidence.end <= len(window.text)
                and (
                    evidence.text is None
                    or window.text[evidence.start:evidence.end]
                    == evidence.text
                )
            ):
                valid.append(evidence)
        if valid:
            evidence_start = window.start + min(item.start for item in valid)
            evidence_end = window.start + max(item.end for item in valid)
        elif judgment.evidence_spans:
            warning = f"LLM 证据位置无效：window_id={window.window_id}"
        return (
            window.model_copy(
                update={
                    "relevance_status": (
                        "relevant" if judgment.relevant else "irrelevant"
                    ),
                    "relevance_type": (
                        judgment.relevance_types[0]
                        if judgment.relevance_types
                        else window.relevance_type
                    ),
                    "evidence_start": evidence_start,
                    "evidence_end": evidence_end,
                    "llm_score": judgment.score,
                    "processing_source": "combined",
                }
            ),
            warning,
        )

    # 中文注释：基于 fast tokenizer offset 构建句子优先的 token 区间，并控制最大 token 数。
    def _token_ranges(
        self,
        text: str,
        range_start: int,
        range_end: int,
        sentences: list[TextSegment],
    ) -> list[tuple[int, int, int | None]]:
        pieces = [
            (max(item.start, range_start), min(item.end, range_end))
            for item in sentences
            if item.start < range_end and item.end > range_start
        ] or [(range_start, range_end)]
        output: list[tuple[int, int, int | None]] = []
        index = 0
        while index < len(pieces):
            start = pieces[index][0]
            first_end = pieces[index][1]
            first_count = self._token_count(text[start:first_end])
            if first_count > self.config.model_max_tokens:
                output.extend(self._split_long_token_range(text, start, first_end))
                index += 1
                continue
            end = first_end
            count = first_count
            next_index = index + 1
            while next_index < len(pieces):
                candidate_end = pieces[next_index][1]
                candidate_count = self._token_count(text[start:candidate_end])
                if candidate_count > self.config.model_max_tokens:
                    break
                end, count = candidate_end, candidate_count
                next_index += 1
            output.append((start, end, count))
            if next_index >= len(pieces):
                break
            overlap_index = next_index
            while overlap_index > index + 1:
                candidate_start = pieces[overlap_index - 1][0]
                if self._token_count(text[candidate_start:end]) >= (
                    self.config.model_overlap_tokens
                ):
                    overlap_index -= 1
                    break
                overlap_index -= 1
            index = (
                overlap_index
                if overlap_index > index
                else index + 1
            )
        return output

    def _split_long_token_range(
        self, text: str, start: int, end: int
    ) -> list[tuple[int, int, int | None]]:
        encoded = self.tokenizer(
            text[start:end],
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = encoded.get("offset_mapping")
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if (
            not offsets
            or not isinstance(offsets, Sequence)
            or isinstance(offsets[0][0], Sequence)
        ):
            raise ModelChunkingError(
                "tokenizer 未返回可用的 offset_mapping，无法安全切分长句"
            )
        valid = [
            (int(item[0]), int(item[1]))
            for item in offsets
            if int(item[1]) > int(item[0])
        ]
        step = max(
            1, self.config.model_max_tokens - self.config.model_overlap_tokens
        )
        ranges: list[tuple[int, int, int | None]] = []
        token_index = 0
        while token_index < len(valid):
            batch = valid[
                token_index:token_index + self.config.model_max_tokens
            ]
            chunk_start = start + batch[0][0]
            chunk_end = start + batch[-1][1]
            ranges.append((chunk_start, chunk_end, len(batch)))
            if token_index + self.config.model_max_tokens >= len(valid):
                break
            token_index += step
        return ranges

    # 中文注释：tokenizer 不可用时使用固定字符窗口和比例重叠生成回退区间。
    def _fallback_ranges(
        self, start: int, end: int
    ) -> list[tuple[int, int, int | None]]:
        maximum = self.config.model_fallback_max_chars
        overlap = min(
            maximum - 1,
            round(
                maximum
                * self.config.model_overlap_tokens
                / self.config.model_max_tokens
            ),
        )
        step = max(1, maximum - overlap)
        ranges = []
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + maximum)
            ranges.append((cursor, chunk_end, None))
            if chunk_end == end:
                break
            cursor += step
        return ranges

    def _expanded_ranges(
        self,
        spans: list[RelevantSpan],
        sentences: list[TextSegment],
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for span in spans:
            indexes = [
                index
                for index, sentence in enumerate(sentences)
                if sentence.start < span.end and sentence.end > span.start
            ]
            if not indexes:
                ranges.append((span.start, span.end))
                continue
            before = max(
                self.config.evidence_before_sentences,
                self.config.context_sentences,
            )
            after = max(
                self.config.evidence_after_sentences,
                self.config.context_sentences,
            )
            first = max(0, min(indexes) - before)
            last = min(len(sentences) - 1, max(indexes) + after)
            ranges.append((sentences[first].start, sentences[last].end))
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    # 中文注释：仅从已配置的本地 checkpoint 懒加载 fast tokenizer，避免隐式联网下载。
    def _ensure_tokenizer(self) -> Any:
        if self.tokenizer is not None or self._tokenizer_attempted:
            return self.tokenizer
        self._tokenizer_attempted = True
        if not self.config.tokenizer_from_ner_model:
            self._tokenizer_warning = (
                "未启用 NER tokenizer，模型块使用字符回退策略"
            )
            return None
        checkpoint = (
            self.project_config.training.get("modeling", {})
            .get("ner", {})
            .get("checkpoint_path")
        )
        if not checkpoint or not Path(checkpoint).is_dir():
            self._tokenizer_warning = (
                "NER tokenizer 本地路径不可用，模型块使用字符回退策略"
            )
            return None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                use_fast=True,
                local_files_only=True,
            )
            if not getattr(tokenizer, "is_fast", False):
                raise ValueError("NER tokenizer 不是 fast tokenizer")
            self.tokenizer = tokenizer
        except (ImportError, OSError, ValueError) as exc:
            self._tokenizer_warning = (
                f"NER tokenizer 加载失败，模型块使用字符回退策略："
                f"{type(exc).__name__}"
            )
        return self.tokenizer

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        values = encoded["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], Sequence):
            values = values[0]
        return len(values)

    # 中文注释：读取并渲染相关性 Prompt；这是当前四个 Prompt 中唯一的生产运行时调用点。
    def _render_relevance_prompt(
        self, document: RawDocument, text: str
    ) -> str:
        if self._prompt_template is None:
            prompt_config = self.project_config.workflow.get(
                "prompts", {}
            ).get("relevance_filter", {})
            configured = str(
                prompt_config.get(
                    "path", "prompts/relevance_filter_prompt.jinja2"
                )
            )
            path = Path(configured)
            if not path.is_absolute():
                path = BASE_DIR / path
            if not path.is_file():
                raise RelevanceServiceError(
                    f"相关性 Prompt 不存在：{path}"
                )
            self._prompt_template = path.read_text(encoding="utf-8")
        values = {
            "source_name": document.source_id,
            "title": document.title or "",
            "published_at": (
                document.published_at.isoformat()
                if document.published_at
                else ""
            ),
            "url": document.source_url or "",
            "text": text,
        }
        rendered = self._prompt_template
        for name, value in values.items():
            rendered = re.sub(
                r"{{\s*" + re.escape(name) + r"\s*}}",
                lambda _: value,
                rendered,
            )
        return rendered

    def _detect_content_type(self, document: RawDocument) -> str:
        declared = (document.content_type or "").split(";", 1)[0].strip().lower()
        extension = (
            Path(document.local_path).suffix.lower()
            if document.local_path
            else ""
        )
        extension_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".pdf": "application/pdf",
            ".docx": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
        }
        if declared:
            return declared
        if extension in extension_types:
            return extension_types[extension]
        sample = (document.raw_text or "").lstrip().lower()
        if sample.startswith("<!doctype html") or "<html" in sample[:500]:
            return "text/html"
        guessed = (
            mimetypes.guess_type(document.local_path)[0]
            if document.local_path
            else None
        )
        return guessed or "text/plain"

    @staticmethod
    def _read_file(path: str | None, content_type: str) -> str:
        if not path:
            raise DocumentParseError("local_path 为空")
        target = Path(path)
        if not target.is_file():
            raise DocumentParseError(f"文档文件不存在：{target}")
        if content_type not in {"text/plain", "text/markdown", "text/html"}:
            return ""
        data = target.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise DocumentParseError(f"无法识别文本编码：{target}")

    def _parse_docx(
        self, path: Path, document: RawDocument
    ) -> ParseResult:
        if not path.is_file():
            raise DocumentParseError(f"DOCX 文件不存在：{path}")
        with zipfile.ZipFile(path) as archive:
            try:
                root = ElementTree.fromstring(
                    archive.read("word/document.xml")
                )
            except KeyError as exc:
                raise DocumentParseError(
                    f"DOCX 缺少 word/document.xml：{path}"
                ) from exc
        namespace = {
            "w": (
                "http://schemas.openxmlformats.org/"
                "wordprocessingml/2006/main"
            )
        }
        paragraphs = [
            "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
            for paragraph in root.findall(".//w:p", namespace)
        ]
        return ParseResult(
            text="\n\n".join(item for item in paragraphs if item),
            title=document.title,
            published_at=document.published_at,
            metadata={**document.metadata, "content_type": document.content_type},
            parser_name="stdlib_docx",
            parser_version=self.parser_version,
        )

    def _parse_pdf(
        self, path: Path, document: RawDocument
    ) -> ParseResult:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise UnsupportedDocumentError(
                "PDF 文本层解析需要可选依赖 pypdf；当前未安装，且本服务不提供 OCR"
            ) from exc
        if not path.is_file():
            raise DocumentParseError(f"PDF 文件不存在：{path}")
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            raise DocumentParseError(f"PDF 文本层解析失败：{path}") from exc
        text = "\n\n".join(pages)
        warnings = [] if text.strip() else ["PDF 没有可提取文本层，可能需要 OCR"]
        return ParseResult(
            text=text,
            title=document.title,
            published_at=document.published_at,
            metadata={
                **document.metadata,
                "content_type": "application/pdf",
                "page_count": len(pages),
            },
            parser_name="pypdf",
            parser_version=getattr(
                __import__("pypdf"), "__version__", None
            ),
            warnings=warnings,
        )

    @staticmethod
    def _remove_decorative_lines(
        chars: list[str], origins: list[list[int]]
    ) -> tuple[list[str], list[list[int]]]:
        text = "".join(chars)
        keep = [True] * len(chars)
        navigation = re.compile(
            r"^\s*(首页|上一页|下一页|返回顶部|相关推荐|责任编辑)\s*$"
        )
        decoration = re.compile(r"^\s*[-_=·•◆◇■□*]{3,}\s*$")
        for match in re.finditer(r"[^\n]*(?:\n|$)", text):
            line = match.group(0).rstrip("\n")
            if navigation.match(line) or decoration.match(line):
                for index in range(match.start(), match.end()):
                    keep[index] = False
        return (
            [char for index, char in enumerate(chars) if keep[index]],
            [origin for index, origin in enumerate(origins) if keep[index]],
        )

    @staticmethod
    def _append_paragraph(
        output: list[TextSegment], text: str, start: int, end: int
    ) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            return
        segment_id = _stable_id("paragraph", start, end, text[start:end])
        output.append(
            TextSegment(
                segment_id=segment_id,
                segment_type="paragraph",
                text=text[start:end],
                start=start,
                end=end,
                paragraph_id=segment_id,
                order=len(output),
            )
        )

    @staticmethod
    def _append_sentence(
        output: list[TextSegment],
        text: str,
        start: int,
        end: int,
        paragraph: TextSegment,
    ) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            return
        segment_id = _stable_id("sentence", start, end, text[start:end])
        output.append(
            TextSegment(
                segment_id=segment_id,
                segment_type="sentence",
                text=text[start:end],
                start=start,
                end=end,
                paragraph_id=paragraph.segment_id,
                sentence_id=segment_id,
                order=len(output),
            )
        )

    def _window_pieces(
        self, sentences: list[TextSegment]
    ) -> list[tuple[int, int]]:
        pieces: list[tuple[int, int]] = []
        maximum = self.config.relevance_max_chars
        overlap = min(
            self.config.relevance_overlap_chars, maximum - 1
        )
        step = max(1, maximum - overlap)
        for sentence in sentences:
            if sentence.end - sentence.start <= maximum:
                pieces.append((sentence.start, sentence.end))
                continue
            cursor = sentence.start
            while cursor < sentence.end:
                end = min(sentence.end, cursor + maximum)
                pieces.append((cursor, end))
                if end == sentence.end:
                    break
                cursor += step
        return pieces

    @staticmethod
    def _validate_windows(
        windows: Sequence[TextWindow], text: str
    ) -> None:
        for window in windows:
            if (
                window.end > len(text)
                or text[window.start:window.end] != window.text
            ):
                raise InvalidTextWindowError(
                    f"窗口字符位置无效：{window.window_id}"
                )

    @staticmethod
    def _deduplicate_chunks(
        chunks: list[ModelInputChunk],
    ) -> list[ModelInputChunk]:
        unique = {
            (item.start, item.end): item
            for item in sorted(chunks, key=lambda value: (value.start, value.end))
        }
        return list(unique.values())

    def _validate_config(self) -> None:
        if self.config.relevance_target_chars > self.config.relevance_max_chars:
            raise ValueError("relevance target_chars 不能大于 max_chars")
        if (
            self.config.relevance_overlap_chars
            >= self.config.relevance_max_chars
        ):
            raise ValueError("relevance overlap_chars 必须小于 max_chars")
        if self.config.reject_threshold > self.config.accept_threshold:
            raise ValueError("reject_threshold 不能大于 accept_threshold")
        if self.config.model_overlap_tokens >= self.config.model_max_tokens:
            raise ValueError("model overlap_tokens 必须小于 max_tokens")

    @staticmethod
    def _processed_case(
        document: RawDocument,
        parsed: ParseResult,
        cleaned: CleanTextResult,
        case_id: str,
        paragraphs: list[TextSegment],
        sentences: list[TextSegment],
        windows: list[TextWindow],
        spans: list[RelevantSpan],
        chunks: list[ModelInputChunk],
        status: Literal["ready", "irrelevant", "partial", "failed"],
        errors: list[str],
    ) -> ProcessedCase:
        return ProcessedCase(
            case_id=case_id,
            doc_id=document.doc_id,
            doc_version_id=document.doc_version_id,
            title=parsed.title or document.title,
            source_id=document.source_id,
            source_url=document.source_url,
            published_at=parsed.published_at or document.published_at,
            original_text=parsed.text,
            cleaned_text=cleaned.text,
            paragraphs=paragraphs,
            sentences=sentences,
            windows=windows,
            relevant_spans=spans,
            model_input_chunks=chunks,
            original_to_cleaned_mapping=cleaned.original_to_cleaned_mapping,
            cleaned_to_original_mapping=cleaned.cleaned_to_original_mapping,
            processing_status=status,
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            processing_errors=errors,
        )

    @staticmethod
    def _failed_case(
        document: RawDocument, exc: Exception
    ) -> ProcessedCase:
        return ProcessedCase(
            case_id=_stable_id(
                "case", document.doc_id, document.doc_version_id or ""
            ),
            doc_id=document.doc_id,
            doc_version_id=document.doc_version_id,
            title=document.title,
            source_id=document.source_id,
            source_url=document.source_url,
            published_at=document.published_at,
            original_text=document.raw_text or "",
            cleaned_text="",
            processing_status="failed",
            parser_name="failed",
            processing_errors=[f"{type(exc).__name__}: {exc}"],
        )

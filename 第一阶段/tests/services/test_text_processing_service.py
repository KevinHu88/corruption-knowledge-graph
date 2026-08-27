"""TextProcessingService 的离线字符偏移、相关性和分块测试。"""

from __future__ import annotations

import asyncio
import zipfile

import pytest

from config import load_project_config
from models import (
    RawDocument,
    RelevantSpan,
    RelevanceJudgment,
    TextWindow,
)
from src.services.text_processing_service import (
    EmptyDocumentError,
    TextProcessingConfig,
    TextProcessingService,
    UnsupportedDocumentError,
)


class MockTokenizer:
    """以单个字符模拟 token，同时提供标准 offset_mapping。"""

    is_fast = True

    def __call__(self, text, **kwargs):
        result = {"input_ids": list(range(len(text)))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class MockLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def generate_structured(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        if self.error:
            raise self.error
        return self.result

    def generate_structured_response(self, *args, **kwargs):
        return self.generate_structured(*args, **kwargs)


def make_config(**updates):
    base = TextProcessingConfig.from_project(load_project_config())
    defaults = {
        "minimum_text_length": 1,
        "relevance_target_chars": 30,
        "relevance_max_chars": 45,
        "relevance_overlap_chars": 8,
        "accept_threshold": 0.7,
        "reject_threshold": 0.2,
        "positive_terms": {"请托": 0.8, "模糊线索": 0.4},
        "negative_terms": {"政策宣传": 0.8},
        "span_max_gap_chars": 2,
        "evidence_before_sentences": 0,
        "evidence_after_sentences": 0,
        "context_sentences": 0,
        "model_max_tokens": 20,
        "model_overlap_tokens": 5,
        "model_fallback_max_chars": 20,
    }
    defaults.update(updates)
    return base.model_copy(update=defaults)


def document(text, **updates):
    values = {
        "doc_id": "doc-1",
        "doc_version_id": "v1",
        "source_id": "court",
        "content_type": "text/plain",
        "raw_text": text,
    }
    values.update(updates)
    return RawDocument(**values)


def assert_all_slices(case):
    for item in [
        *case.paragraphs,
        *case.sentences,
        *case.windows,
        *case.relevant_spans,
        *case.model_input_chunks,
    ]:
        assert case.cleaned_text[item.start:item.end] == item.text


def test_clean_text_normalizes_crlf_zero_width_and_spaces_with_mapping():
    service = TextProcessingService(config=make_config())
    original = "甲\r\n\u200b乙  \t丙"

    result = service.clean_text(original)

    assert result.text == "甲\n乙 丙"
    assert len(result.cleaned_to_original_mapping) == len(result.text)
    for clean_index, original_index in enumerate(
        result.cleaned_to_original_mapping
    ):
        expected = result.text[clean_index]
        actual = original[original_index]
        if expected == "\n":
            assert actual == "\r"
        elif expected == " ":
            assert actual in {" ", "\t"}
        else:
            assert actual == expected


def test_repeated_sentences_keep_sequential_offsets_and_paragraph_slices():
    service = TextProcessingService(config=make_config())
    text = "重复句子。重复句子。\n\n第二段。"
    paragraphs = service.split_paragraphs(text)
    sentences = service.split_sentences(text, paragraphs)

    assert sentences[0].start == 0
    assert sentences[1].start == len("重复句子。")
    assert sentences[0].text == sentences[1].text
    assert all(text[item.start:item.end] == item.text for item in paragraphs)
    assert all(text[item.start:item.end] == item.text for item in sentences)


def test_sentence_splitter_keeps_decimal_and_common_abbreviation():
    service = TextProcessingService(config=make_config())
    text = "金额为3.5万元。Dr. Smith approved. 下一句。"
    paragraphs = service.split_paragraphs(text)
    sentences = service.split_sentences(text, paragraphs)

    assert sentences[0].text == "金额为3.5万元。"
    assert sentences[1].text == "Dr. Smith approved."
    assert sentences[2].text == "下一句。"


def test_html_and_docx_are_parsed_without_third_party_dependencies(tmp_path):
    service = TextProcessingService(config=make_config())
    html_result = service.parse_document(
        document(
            "<html><head><title>标题</title></head>"
            "<body><nav>导航</nav><p>案件正文。</p></body></html>",
            content_type="text/html",
            title=None,
        )
    )
    assert html_result.title == "标题"
    assert "案件正文" in html_result.text
    assert "导航" not in html_result.text

    path = tmp_path / "sample.docx"
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>'
        '<w:p><w:r><w:t>第一段</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>第二段</w:t></w:r></w:p>'
        '</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    docx_result = service.parse_document(
        RawDocument(
            doc_id="docx-1",
            source_id="local",
            local_path=str(path),
            content_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
        )
    )
    assert docx_result.text == "第一段\n\n第二段"


def test_plain_text_generates_processed_case_and_all_global_slices():
    mock = MockLLM()
    service = TextProcessingService(
        config=make_config(), llm_service=mock, tokenizer=MockTokenizer()
    )

    case = asyncio.run(service.process_document(
        document("张某任某局局长。李某请托张某帮助公司承揽项目。")
    ))

    assert case.processing_status == "ready"
    assert case.model_input_chunks
    assert mock.calls == 0
    assert_all_slices(case)


def test_clearly_irrelevant_text_does_not_call_llm():
    mock = MockLLM()
    service = TextProcessingService(
        config=make_config(), llm_service=mock, tokenizer=MockTokenizer()
    )

    case = asyncio.run(
        service.process_document(document("这是一段政策宣传材料。"))
    )

    assert case.processing_status == "irrelevant"
    assert not case.relevant_spans
    assert mock.calls == 0


def test_uncertain_window_calls_llm_and_uses_full_window_span():
    text = "开头说明。这里有模糊线索需要判断。结尾。"
    local_text = "模糊线索"
    local_start = text.index(local_text)
    mock = MockLLM(
        RelevanceJudgment(
            relevant=True,
            score=0.86,
            relevance_types=["请托关系"],
            evidence_spans=[
                {
                    "start": local_start,
                    "end": local_start + len(local_text),
                    "text": local_text,
                }
            ],
            reason="存在具体线索",
        )
    )
    service = TextProcessingService(
        config=make_config(relevance_max_chars=100),
        llm_service=mock,
        tokenizer=MockTokenizer(),
    )

    case = asyncio.run(service.process_document(document(text)))

    window = case.windows[0]
    assert mock.calls == 1
    assert window.evidence_start is None
    assert window.evidence_end is None
    assert window.processing_source == "combined"
    assert case.relevant_spans[0].text == window.text


def test_transport_ignores_nested_llm_evidence_fields():
    mock = MockLLM(
        RelevanceJudgment(
            relevant=True,
            score=0.8,
            relevance_types=["其他"],
            evidence_spans=[{"start": 0, "end": 999, "text": None}],
        )
    )
    service = TextProcessingService(
        config=make_config(),
        llm_service=mock,
        tokenizer=MockTokenizer(),
    )

    case = asyncio.run(
        service.process_document(document("存在模糊线索。"))
    )

    assert case.windows[0].evidence_start is None
    assert case.windows[0].evidence_end is None
    assert not any("证据位置无效" in item for item in case.processing_errors)
    assert case.processing_status == "ready"


def test_llm_failure_keeps_document_as_partial():
    service = TextProcessingService(
        config=make_config(),
        llm_service=MockLLM(error=RuntimeError("offline")),
        tokenizer=MockTokenizer(),
    )

    case = asyncio.run(
        service.process_document(document("存在模糊线索。"))
    )

    assert case.processing_status == "partial"
    assert case.windows[0].relevance_status == "uncertain"
    assert any("LLM 相关性判断失败" in item for item in case.processing_errors)


def test_relevant_span_merge_respects_overlap_and_max_gap():
    service = TextProcessingService(config=make_config(span_max_gap_chars=2))
    text = "甲乙丙丁戊己庚辛壬癸"
    windows = [
        TextWindow(
            window_id="w1", text=text[0:3], start=0, end=3,
            segment_ids=[], rule_score=0.8,
            relevance_status="relevant", processing_source="rule",
        ),
        TextWindow(
            window_id="w2", text=text[2:5], start=2, end=5,
            segment_ids=[], rule_score=0.9,
            relevance_status="relevant", processing_source="rule",
        ),
        TextWindow(
            window_id="w3", text=text[8:10], start=8, end=10,
            segment_ids=[], rule_score=0.8,
            relevance_status="relevant", processing_source="rule",
        ),
    ]

    spans = service.merge_relevant_spans(windows, text)

    assert [(item.start, item.end) for item in spans] == [(0, 5), (8, 10)]
    assert all(text[item.start:item.end] == item.text for item in spans)


def test_model_chunks_respect_tokens_boundaries_and_overlap_long_sentence():
    service = TextProcessingService(
        config=make_config(model_max_tokens=10, model_overlap_tokens=3),
        tokenizer=MockTokenizer(),
    )
    text = "甲" * 27 + "。"
    paragraphs = service.split_paragraphs(text)
    sentences = service.split_sentences(text, paragraphs)
    span = RelevantSpan(
        span_id="s1",
        text=text,
        start=0,
        end=len(text),
        source_window_ids=["w1"],
        relevance_types=["test"],
        score=0.9,
    )

    chunks = service.build_model_input_chunks(
        "case-1", text, [span], sentences
    )

    assert len(chunks) > 1
    assert all(item.token_count <= 10 for item in chunks)
    assert all(text[item.start:item.end] == item.text for item in chunks)
    assert any(item.overlap_left >= 3 for item in chunks[1:])


def test_chunk_start_converts_local_entities_to_full_text_offsets():
    text = (
        "张某曾任某局局长。"
        "李某请托张某帮助某公司承揽项目，后向张某支付人民币二十万元。"
    )
    service = TextProcessingService(
        config=make_config(
            positive_terms={
                "请托": 0.4,
                "承揽": 0.2,
                "支付": 0.2,
            },
            relevance_max_chars=100,
            relevance_target_chars=80,
            model_max_tokens=100,
        ),
        tokenizer=MockTokenizer(),
    )

    case = asyncio.run(service.process_document(document(text)))
    chunk = case.model_input_chunks[0]
    for name in ["张某", "李某", "某公司", "人民币二十万元"]:
        local_start = chunk.text.index(name)
        local_end = local_start + len(name)
        global_start = chunk.start + local_start
        global_end = chunk.start + local_end
        assert case.cleaned_text[global_start:global_end] == name


def test_empty_and_unsupported_documents_raise_clear_errors():
    service = TextProcessingService(config=make_config())
    with pytest.raises(EmptyDocumentError):
        asyncio.run(service.process_document(document("\u200b \r\n")))
    with pytest.raises(UnsupportedDocumentError):
        asyncio.run(service.process_document(
            document("binary", content_type="application/octet-stream")
        ))


def test_batch_preserves_order_and_contains_individual_failure():
    service = TextProcessingService(
        config=make_config(), tokenizer=MockTokenizer()
    )
    results = asyncio.run(service.process_documents(
        [
            document(
                "李某请托张某办理事项。",
                doc_id="ok",
                doc_version_id="1",
            ),
            document(
                "",
                doc_id="bad",
                doc_version_id="1",
            ),
        ]
    ))

    assert [item.doc_id for item in results] == ["ok", "bad"]
    assert results[0].processing_status == "ready"
    assert results[1].processing_status == "failed"

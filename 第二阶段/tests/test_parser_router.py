from pypdf import PdfWriter
from docx import Document

from 第二阶段.parsing.parsers.docx_parser import DocxParser
from 第二阶段.parsing.parsers.pdf_parser import PDFParser
from 第二阶段.parsing.parsers.txt_parser import TxtParser
from 第二阶段.parsing.router import ParserRouter


def test_txt_routes_to_txt_parser_and_parses(tmp_path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("张某与李某存在请托关系。", encoding="utf-8")
    router = ParserRouter()
    assert isinstance(router.resolve_parser(path), TxtParser)
    parsed = router.parse(path)
    assert parsed.file_type == "txt"
    assert "请托关系" in parsed.text


def test_pdf_parser_works_for_valid_pdf(tmp_path) -> None:
    path = tmp_path / "example.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)
    parsed = ParserRouter().parse(path)
    assert isinstance(ParserRouter().resolve_parser(path), PDFParser)
    assert parsed.metadata["page_count"] == 1


def test_docx_parser_extracts_paragraphs(tmp_path) -> None:
    path = tmp_path / "example.docx"
    document = Document()
    document.add_paragraph("项目审批材料")
    document.save(path)
    parsed = ParserRouter().parse(path)
    assert isinstance(ParserRouter().resolve_parser(path), DocxParser)
    assert "项目审批材料" in parsed.text


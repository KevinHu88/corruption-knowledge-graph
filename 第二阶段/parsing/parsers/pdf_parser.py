"""PDF 文本解析器。"""

from pathlib import Path

from 第二阶段.parsing.base import BaseParser, ParserDependencyError, ParserError
from 第二阶段.schemas.models import ParsedDocument


class PDFParser(BaseParser):
    file_type = "pdf"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = self._validate_path(file_path)
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ParserDependencyError("PDF 解析需要安装 pypdf") from exc
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
        except Exception as exc:
            raise ParserError(f"PDF 解析失败：{path}") from exc
        return self._build_document(
            path,
            "\n\n".join(text for text in pages if text),
            metadata={"page_count": len(reader.pages)},
        )


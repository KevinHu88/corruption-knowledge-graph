"""根据扩展名与 MIME 类型选择解析器。"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from 第二阶段.parsing.base import BaseParser, UnsupportedFileTypeError
from 第二阶段.parsing.parsers import (
    DocxParser,
    ExcelParser,
    HtmlParser,
    PDFParser,
    PptxParser,
    TxtParser,
)
from 第二阶段.schemas.models import ParsedDocument


class ParserRouter:
    """所有文件类型判断的唯一入口。"""

    def __init__(self) -> None:
        self._extension_parsers: dict[str, BaseParser] = {
            ".txt": TxtParser(),
            ".pdf": PDFParser(),
            ".docx": DocxParser(),
            ".html": HtmlParser(),
            ".htm": HtmlParser(),
            ".xlsx": ExcelParser(),
            ".pptx": PptxParser(),
        }
        self._mime_parsers: dict[str, BaseParser] = {
            "text/plain": self._extension_parsers[".txt"],
            "application/pdf": self._extension_parsers[".pdf"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._extension_parsers[".docx"],
            "text/html": self._extension_parsers[".html"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": self._extension_parsers[".xlsx"],
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": self._extension_parsers[".pptx"],
        }

    def resolve_parser(
        self, file_path: str | Path, mime_type: str | None = None
    ) -> BaseParser:
        path = Path(file_path)
        parser = self._extension_parsers.get(path.suffix.lower())
        detected_mime = mime_type or mimetypes.guess_type(path.name)[0]
        if parser is None and detected_mime:
            parser = self._mime_parsers.get(detected_mime.split(";", 1)[0].strip())
        if parser is None:
            raise UnsupportedFileTypeError(
                f"不支持的文件类型：extension={path.suffix or '<none>'}, mime={detected_mime}"
            )
        return parser

    def parse(
        self, file_path: str | Path, mime_type: str | None = None
    ) -> ParsedDocument:
        return self.resolve_parser(file_path, mime_type).parse(file_path)


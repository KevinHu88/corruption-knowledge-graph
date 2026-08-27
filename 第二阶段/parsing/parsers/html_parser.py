"""HTML 可见文本解析器。"""

from pathlib import Path

from 第二阶段.parsing.base import BaseParser, ParserDependencyError
from 第二阶段.parsing.parsers.txt_parser import TxtParser
from 第二阶段.schemas.models import ParsedDocument


class HtmlParser(BaseParser):
    file_type = "html"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = self._validate_path(file_path)
        raw = TxtParser().parse(path).text
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ParserDependencyError("HTML 解析需要安装 beautifulsoup4") from exc
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return self._build_document(path, soup.get_text("\n", strip=True))


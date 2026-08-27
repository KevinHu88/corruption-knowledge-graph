"""纯文本解析器。"""

from pathlib import Path

from 第二阶段.parsing.base import BaseParser, ParserError
from 第二阶段.schemas.models import ParsedDocument


class TxtParser(BaseParser):
    file_type = "txt"

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = self._validate_path(file_path)
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                text = path.read_text(encoding=encoding)
                return self._build_document(
                    path, text, metadata={"encoding": encoding}
                )
            except UnicodeDecodeError:
                continue
        raise ParserError(f"无法识别文本编码：{path}")


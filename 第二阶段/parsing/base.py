"""解析器抽象接口与通用校验。"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from 第二阶段.schemas.models import ParsedDocument


class ParserError(RuntimeError):
    """文件解析失败。"""


class ParserDependencyError(ParserError):
    """解析器所需的可选依赖未安装。"""


class UnsupportedFileTypeError(ParserError):
    """文件类型不受当前 ParserRouter 支持。"""


class BaseParser(ABC):
    """所有文件解析器必须实现的统一接口。"""

    file_type: str

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        """解析单个文件并返回统一文档。"""

    def _validate_path(self, file_path: str | Path) -> Path:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if not path.is_file():
            raise ParserError(f"不是普通文件：{path}")
        return path

    def _build_document(
        self,
        path: Path,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        stat = path.stat()
        identity = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
        document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return ParsedDocument(
            document_id=document_id,
            file_name=path.name,
            file_type=self.file_type,
            text=text.strip(),
            metadata={
                "file_path": str(path),
                "file_size": stat.st_size,
                **dict(metadata or {}),
            },
        )


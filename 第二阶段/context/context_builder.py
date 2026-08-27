"""将统一 Evidence 格式化为有长度上限的 LLM 上下文。"""

from 第二阶段.schemas.models import Evidence


class ContextBuilder:
    def __init__(self, max_chars: int = 12000) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")
        self.max_chars = max_chars

    def build(self, evidence: list[Evidence]) -> str:
        if not evidence:
            return "未检索到可用证据。"
        blocks: list[str] = []
        used = 0
        for index, item in enumerate(evidence, start=1):
            header = (
                f"[Evidence {index}]\n"
                f"Evidence ID: {item.id}\n"
                f"Source Type: {item.source_type}\n"
                f"Source: {item.source or 'unknown'}\n"
                "Content: "
            )
            remaining = self.max_chars - used - len(header)
            if remaining <= 0:
                break
            content = item.content[:remaining]
            block = header + content
            blocks.append(block)
            used += len(block) + 2
            if len(content) < len(item.content):
                break
        return "\n\n".join(blocks)


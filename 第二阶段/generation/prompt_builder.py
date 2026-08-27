"""知识问答 Prompt 构建。"""


class PromptBuilder:
    def build(self, question: str, context: str) -> str:
        if not question.strip():
            raise ValueError("question 不能为空")
        return (
            "你是合谋腐败知识问答助手。请严格遵守以下规则：\n"
            "1. 优先依据给出的 Evidence 回答，不使用无法验证的外部事实。\n"
            "2. 不确定时明确说明，不得虚构实体、关系或事实。\n"
            "3. 明确区分上传文件证据（document）和知识图谱证据（graph）。\n"
            "4. 引用事实时尽可能标注对应的 Evidence ID。\n"
            "5. 没有足够证据时直接说明“证据不足”。\n\n"
            f"用户问题：\n{question.strip()}\n\n"
            f"Evidence：\n{context}"
        )


"""第二阶段业务层与 HTTP 层共享的可分类异常。"""


class KnowledgeQAError(RuntimeError):
    """第二阶段可预期异常的基类。"""


class SessionNotFoundError(KnowledgeQAError):
    """请求的临时 Session 不存在。"""


class FileTooLargeError(KnowledgeQAError):
    """上传文件超过配置限制。"""


class DocumentParsingError(KnowledgeQAError):
    """已允许类型的文件无法完成解析。"""


class GraphRetrievalError(KnowledgeQAError):
    """知识图谱检索失败。"""


class AmbiguousEntityError(KnowledgeQAError):
    """同名实体分布在多个案件中，需要调用方指定案件范围。"""

    def __init__(self, entity_name: str, candidate_case_ids: list[str]) -> None:
        self.entity_name = entity_name
        self.candidate_case_ids = sorted(set(candidate_case_ids))
        cases = "、".join(self.candidate_case_ids)
        super().__init__(
            f"实体“{entity_name}”存在于多个案件（{cases}），请指定 case_id。"
        )


class LLMGenerationError(KnowledgeQAError):
    """大模型生成失败。"""


class InvalidQuestionError(KnowledgeQAError):
    """问题为空或不符合基础长度要求。"""

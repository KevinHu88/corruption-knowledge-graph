"""可替换为 LLM Router 的规则路由器。"""

from __future__ import annotations

from 第二阶段.schemas.models import QueryPlan


class QueryRouter:
    """根据问题措辞和会话文档状态选择检索来源。"""

    GRAPH_TERMS = (
        "关系",
        "关联",
        "任职",
        "与谁",
        "通过谁",
        "路径",
        "组织",
        "人员网络",
        "请托",
        "合谋",
        "亲属",
        "同事",
        "领导",
        "利益输送",
    )
    DOCUMENT_TERMS = (
        "文件",
        "原文",
        "材料",
        "报告",
        "文中",
        "上传",
        "如何描述",
        "依据",
        "证据",
    )

    def route(self, question: str, has_uploaded_documents: bool) -> QueryPlan:
        normalized = question.strip()
        if not normalized:
            raise ValueError("question 不能为空")
        graph_hits = [term for term in self.GRAPH_TERMS if term in normalized]
        document_hits = [term for term in self.DOCUMENT_TERMS if term in normalized]
        reasons: list[str] = []

        if not has_uploaded_documents:
            reasons.append("当前会话没有上传文档，只检索知识图谱")
            return QueryPlan("GRAPH", normalized, reasons)
        if graph_hits and document_hits:
            reasons.append(f"命中文档词：{', '.join(document_hits)}")
            reasons.append(f"命中图谱词：{', '.join(graph_hits)}")
            return QueryPlan("HYBRID", normalized, reasons)
        if document_hits:
            reasons.append(f"命中文档词：{', '.join(document_hits)}")
            return QueryPlan("DOCUMENT", normalized, reasons)
        if graph_hits:
            reasons.append(f"命中图谱词：{', '.join(graph_hits)}")
            return QueryPlan("GRAPH", normalized, reasons)

        reasons.append("问题未命中特定规则，同时检索会话文档和知识图谱")
        return QueryPlan("HYBRID", normalized, reasons)


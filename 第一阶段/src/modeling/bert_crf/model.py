"""Legacy 兼容的 BERT、Linear 与线性链 CRF 网络结构。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel


# 中文注释：线性链 CRF 层，负责序列标注的转移得分、对数似然和 Viterbi 解码。
class LinearChainCRF(nn.Module):
    """与 legacy CRF 参数命名兼容的 batch-first 线性链 CRF。"""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """按 legacy 实现使用 [-0.1, 0.1] 均匀分布初始化。"""

        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 batch 平均条件对数似然。"""

        mask = (
            torch.ones_like(tags, dtype=torch.bool)
            if mask is None
            else mask.bool()
        )
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for index in range(1, emissions.size(1)):
            active = mask[:, index]
            transition = self.transitions[
                tags[:, index - 1], tags[:, index]
            ]
            emission = emissions[:, index].gather(
                1, tags[:, index:index + 1]
            ).squeeze(1)
            score += (transition + emission) * active
        lengths = mask.long().sum(1) - 1
        score += self.end_transitions[tags.gather(1, lengths[:, None]).squeeze(1)]

        normalizer = self.start_transitions + emissions[:, 0]
        for index in range(1, emissions.size(1)):
            candidate = (
                normalizer.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, index].unsqueeze(1)
            )
            next_score = torch.logsumexp(candidate, dim=1)
            normalizer = torch.where(
                mask[:, index:index + 1], next_score, normalizer
            )
        normalizer = torch.logsumexp(
            normalizer + self.end_transitions, dim=1
        )
        return (score - normalizer).mean()

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """使用 Viterbi 返回每个 batch 的最优标签路径。"""

        mask = (
            torch.ones(emissions.shape[:2], dtype=torch.bool,
                       device=emissions.device)
            if mask is None
            else mask.bool()
        )
        score = self.start_transitions + emissions[:, 0]
        history: list[torch.Tensor] = []
        for index in range(1, emissions.size(1)):
            candidate = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_path = candidate.max(1)
            best_score += emissions[:, index]
            score = torch.where(mask[:, index:index + 1], best_score, score)
            history.append(best_path)
        score += self.end_transitions
        best_last = score.argmax(1)
        paths: list[list[int]] = []
        for batch_index in range(emissions.size(0)):
            length = int(mask[batch_index].sum().item())
            tag = int(best_last[batch_index].item())
            path = [tag]
            for step in reversed(history[:length - 1]):
                tag = int(step[batch_index, tag].item())
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths


# 中文注释：BERT 编码器与 CRF 解码层组成的命名实体识别模型。
class BertCrfForNer(BertPreTrainedModel):
    """保持 legacy 权重键结构的 BERT-CRF。"""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = LinearChainCRF(config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """计算发射分数；存在标签时返回负对数似然 loss。"""

        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        emissions = self.classifier(self.dropout(hidden))
        if labels is None:
            return (emissions,)
        loss = -self.crf(emissions, labels, attention_mask)
        return loss, emissions

    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """解码最优 BIO 标签路径。"""

        return self.crf.decode(emissions, attention_mask)

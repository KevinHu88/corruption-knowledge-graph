"""保持 legacy BERTEntity 数学结构与权重键的 PyTorch 网络。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import BertModel


# 中文注释：使用实体位置标记从 BERT 隐状态中提取头实体和尾实体表示。
class BertEntityEncoder(nn.Module):
    """取头尾实体起始 marker 隐状态并拼接。"""

    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden = self.bert.config.hidden_size * 2
        self.hidden_size = hidden
        self.linear = nn.Linear(hidden, hidden)

    def forward(
        self,
        token: torch.Tensor,
        att_mask: torch.Tensor,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        """复用 legacy marker 起始位置编码。"""

        hidden = self.bert(
            token, attention_mask=att_mask, return_dict=False
        )[0]
        head_index = pos1.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        tail_index = pos2.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        head = hidden.gather(1, head_index).squeeze(1)
        tail = hidden.gather(1, tail_index).squeeze(1)
        return self.linear(torch.cat([head, tail], dim=1))


# 中文注释：在实体对表示上增加分类层，输出当前 schema 中各关系类别的 logits。
class BertEntityForRelation(nn.Module):
    """legacy SoftmaxNN：BERTEntityEncoder、Dropout、Linear 分类。"""

    def __init__(
        self,
        pretrained_model: str,
        num_relations: int,
    ) -> None:
        super().__init__()
        self.sentence_encoder = BertEntityEncoder(pretrained_model)
        self.fc = nn.Linear(
            self.sentence_encoder.hidden_size, num_relations
        )
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout()

    def forward(self, *inputs: Any) -> torch.Tensor:
        """输出关系分类 logits。"""

        return self.fc(self.drop(self.sentence_encoder(*inputs)))

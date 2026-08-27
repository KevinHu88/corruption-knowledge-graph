"""合谋腐败知识图谱工作流中的统一数据模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

# 1. 通用类型别名

DocumentId = Annotated[
    str,
    Field(description="原始文档唯一编号，例如 doc_000001"),
]

DocumentVersionId = Annotated[
    str,
    Field(description="文档版本唯一编号，用于区分同一网页的不同内容版本"),
]

CaseId = Annotated[
    str,
    Field(description="案件唯一编号，例如 case_000001"),
]

TextId = Annotated[
    str,
    Field(description="案件文本片段唯一编号，例如 text_000001"),
]

AnnotationId = Annotated[
    str,
    Field(description="统一标注记录唯一编号"),
]

EntityId = Annotated[
    str,
    Field(description="当前文本内部的实体唯一编号，例如 e1、e2"),
]

RelationId = Annotated[
    str,
    Field(description="当前文本内部的关系唯一编号，例如 r1、r2"),
]

Confidence = Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
        description="模型置信度，取值范围为0至1",
    ),
]

CharacterStart = Annotated[
    int,
    Field(
        ge=0,
        description="实体或证据在原文中的起始字符位置，采用左闭右开索引",
    ),
]

CharacterEnd = Annotated[
    int,
    Field(
        gt=0,
        description="实体或证据在原文中的结束字符位置，采用左闭右开索引",
    ),
]

FileUri = Annotated[
    str,
    Field(description="本地文件路径、对象存储地址或其他资源地址"),
]

SchemaVersion = Annotated[
    str,
    Field(description="实体和关系标注规范版本，例如 relation_v2.0"),
]

ModelVersionName = Annotated[
    str,
    Field(description="模型版本编号，例如 re_bert_v4"),
]

DatasetVersionName = Annotated[
    str,
    Field(description="数据集版本编号，例如 dataset_corruption_v003"),
]

# 2. 基础枚举
# 中文注释：统一实体类别枚举，是标注、模型、数据集和图谱之间共享的基础协议。
class EntityType(str, Enum):
    """命名实体类型。"""

    PER = "PER"              # 人物
    ORG = "ORG"              # 机构或组织
    POSITION = "POSITION"    # 职位或职务
    MONEY = "MONEY"          # 金额


# 中文注释：统一关系类别枚举，关系方向和合法实体组合仍由 schema.yaml 约束。
class RelationType(str, Enum):
    """合谋腐败知识图谱关系类型。"""

    NO_RELATION = "无关系"               # 候选实体对之间不存在目标关系

    EMPLOYED_BY = "任职于"               # 人物 → 机构
    HOLDS_POSITION = "担任职位"          # 人物 → 职位
    RECEIVES_MONEY = "收受金额"          # 人物 → 金额
    PAYS_MONEY = "支付金额"              # 人物或机构 → 金额
    RELATED_ORG = "关联机构"             # 机构 → 机构
    TAKES_BRIBE_FROM = "受贿于"          # 受贿人 → 行贿人

    FRIEND = "朋友"                      # 人物 → 人物
    RELATIVE = "亲属"                    # 人物 → 人物
    INTIMATE_RELATION = "亲密关系"       # 人物 → 人物
    COLLEAGUE_CLASSMATE = "同事同学"     # 人物 → 人物

    LEADER = "领导"                      # 领导者 → 被领导者
    POSITIONAL_CONSTRAINT = "职务制约"   # 制约者 → 被制约者
    ENTRUST = "请托"                     # 请托人 → 被请托人
    TRANSFER_ENTRUST = "转请托"          # 中间请托人 → 实际被请托人
    INSTRUCT = "指示办事"                # 指示者 → 被指示者
    HELP_PROFIT = "帮助谋利"             # 帮助者 → 受益人

    BENEFIT_TRANSFER = "利益输送"        # 利益提供者 → 利益接受者
    INTERMEDIARY_INTRODUCTION = "中介介绍"  # 介绍者 → 被介绍对象
    ASSIST_IMPLEMENTATION = "协助实施"   # 协助者 → 被协助者
    HOLD_ON_BEHALF = "代收代持"          # 代收代持人 → 实际受益人
    COLLUSION = "合谋"                   # 人物 → 人物


# 中文注释：描述总工作流当前所处阶段，供状态记录和后续断点恢复设计使用。
class WorkflowStage(str, Enum):
    """工作流当前所处阶段。"""

    INGESTION = "ingestion"                  # 数据采集
    PARSING = "parsing"                      # 文档解析与清洗
    RELEVANCE_FILTER = "relevance_filter"    # 相关性筛选
    ANNOTATION = "annotation"                # 实体关系预标注
    REVIEW = "review"                        # 人工审核
    DATASET_BUILDING = "dataset_building"    # 数据集构建
    TRAINING = "training"                    # 模型训练
    EVALUATION = "evaluation"                # 模型评价
    INFERENCE = "inference"                  # 生产推理
    GRAPH_INGESTION = "graph_ingestion"      # Neo4j图谱入库
    QA = "qa"                                # 知识图谱问答


class TaskStatus(str, Enum):
    """通用任务运行状态。"""

    PENDING = "PENDING"                  # 等待执行
    RUNNING = "RUNNING"                  # 正在执行
    SUCCEEDED = "SUCCEEDED"              # 执行成功
    FAILED = "FAILED"                    # 执行失败
    SKIPPED = "SKIPPED"                  # 已跳过
    WAITING_REVIEW = "WAITING_REVIEW"    # 等待人工审核


# 中文注释：标注生命周期状态；数据集阶段会严格区分生成、待审和已审核标注。
class AnnotationStatus(str, Enum):
    """标注数据状态。"""

    GENERATED = "GENERATED"              # 已由模型生成
    VALIDATION_FAILED = "VALIDATION_FAILED"  # 自动校验失败
    PENDING_REVIEW = "PENDING_REVIEW"    # 等待人工审核
    IN_REVIEW = "IN_REVIEW"              # 正在审核
    APPROVED = "APPROVED"                # 审核通过
    REJECTED = "REJECTED"                # 审核拒绝
    EXPORTED = "EXPORTED"                # 已导出为训练格式
    IN_DATASET = "IN_DATASET"            # 已加入某个数据集版本
    TRAINED = "TRAINED"                  # 已用于模型训练


class ReviewDecision(str, Enum):
    """人工审核结论。"""

    APPROVED = "APPROVED"    # 审核通过
    RETURNED = "RETURNED"    # 退回修改
    REJECTED = "REJECTED"    # 不作为有效样本


class ModelRole(str, Enum):
    """模型在注册中心中的角色。"""

    CHALLENGER = "CHALLENGER"    # 候选模型
    CHAMPION = "CHAMPION"        # 当前正式生产模型
    ARCHIVED = "ARCHIVED"        # 已归档模型


class ClaimStatus(str, Enum):
    """图谱关系事实的确认状态。"""

    MODEL_PREDICTED = "MODEL_PREDICTED"  # 模型预测
    HUMAN_VERIFIED = "HUMAN_VERIFIED"    # 人工确认
    REJECTED = "REJECTED"                # 已否定

# 3. 原始文档与案件文本

# 中文注释：表示可追溯的来源文档版本，主要供规范标注和 Neo4j 写入关联原始证据。
class SourceDocument(BaseModel):
    """从网站、PDF或人工导入获得的原始文档记录。"""

    doc_id: DocumentId

    doc_version_id: DocumentVersionId

    source_id: Annotated[
        str,
        Field(description="来源网站编号，例如 ccdi、court、audit"),
    ]

    title: Annotated[
        str,
        Field(description="文档原始标题"),
    ]

    raw_url: Annotated[
        str | None,
        Field(description="采集时获得的原始URL"),
    ] = None

    canonical_url: Annotated[
        str | None,
        Field(description="去除跟踪参数并完成标准化后的URL"),
    ] = None

    published_at: Annotated[
        datetime | None,
        Field(description="来源网站显示的文档发布时间"),
    ] = None

    ingested_at: Annotated[
        datetime,
        Field(description="文档被当前系统采集的时间"),
    ] = Field(default_factory=datetime.now)

    content_hash: Annotated[
        str | None,
        Field(description="清洗后正文的哈希值，用于内容去重"),
    ] = None

    raw_file_uri: Annotated[
        str | None,
        Field(description="原始网页、PDF或扫描件的存储地址"),
    ] = None

    status: Annotated[
        str,
        Field(description="文档当前处理状态"),
    ] = "DISCOVERED"

    metadata: Annotated[
        dict[str, Any],
        Field(description="来源网站特有的扩展元数据"),
    ] = Field(default_factory=dict)


class EvidenceSpan(BaseModel):
    """原始文本中的一段相关证据。"""

    start: CharacterStart
    end: CharacterEnd

    text: Annotated[
        str | None,
        Field(description="证据区间对应的原始文本"),
    ] = None

    relevance_type: Annotated[
        str | None,
        Field(description="证据类型，例如请托关系、利益输送、人物任职"),
    ] = None

    confidence: Annotated[
        Confidence | None,
        Field(description="相关性判断置信度"),
    ] = None

    @model_validator(mode="after")
    def validate_span(self) -> "EvidenceSpan":
        """检查证据起止位置是否合法。"""

        if self.end <= self.start:
            raise ValueError("证据区间必须满足 end > start")

        return self


# 中文注释：案件级业务对象，为文档、标注、数据集切分和知识图谱提供共同的 case_id。
class CaseDocument(BaseModel):
    """经过解析、清洗和相关性筛选后的案件级文本。"""

    case_id: CaseId
    doc_id: DocumentId
    doc_version_id: DocumentVersionId

    title: Annotated[
        str,
        Field(description="案件材料标题"),
    ]

    source_id: Annotated[
        str,
        Field(description="案件材料来源网站编号"),
    ]

    published_at: Annotated[
        datetime | None,
        Field(description="案件材料发布时间"),
    ] = None

    raw_text: Annotated[
        str,
        Field(description="从原始文件中提取的完整原文"),
    ]

    clean_text: Annotated[
        str,
        Field(description="经过清洗、去噪和字符规范化后的文本"),
    ]

    relevant_spans: Annotated[
        list[EvidenceSpan],
        Field(description="从全文中筛选出的合谋腐败相关证据片段"),
    ] = Field(default_factory=list)

    status: Annotated[
        str,
        Field(description="案件文本当前状态"),
    ] = "READY_FOR_ANNOTATION"

    metadata: Annotated[
        dict[str, Any],
        Field(description="案件编号、地区、案由等附加信息"),
    ] = Field(default_factory=dict)

    # 4. 实体、关系与统一标注
# =========================================================

# 中文注释：规范实体提及，字符区间采用左闭右开约定，并校验文本与 offset 一致性。
class EntityMention(BaseModel):
    """某个实体在具体文本中的一次出现。"""

    entity_id: EntityId

    name: Annotated[
        str,
        Field(description="实体在原始文本中的完整名称"),
    ]

    type: Annotated[
        EntityType,
        Field(description="实体类型：人物、机构、职位或金额"),
    ]

    start: CharacterStart
    end: CharacterEnd

    confidence: Annotated[
        Confidence | None,
        Field(description="实体识别模型或大模型给出的置信度"),
    ] = None

    normalized_name: Annotated[
        str | None,
        Field(description="实体规范化名称，例如机构全称"),
    ] = None

    @model_validator(mode="after")
    def validate_position(self) -> "EntityMention":
        """检查实体字符区间是否合法。"""

        if self.end <= self.start:
            raise ValueError("实体位置必须满足 end > start")

        return self


# 中文注释：规范关系提及，通过实体 ID 连接头尾实体，并可携带证据文本范围。
class RelationMention(BaseModel):
    """当前文本中的一条实体关系。"""

    relation_id: RelationId

    head_id: Annotated[
        str,
        Field(description="头实体ID，必须对应entities中的实体"),
    ]

    tail_id: Annotated[
        str,
        Field(description="尾实体ID，必须对应entities中的实体"),
    ]

    type: Annotated[
        RelationType,
        Field(description="关系类型，例如请托、利益输送或任职于"),
    ]

    confidence: Annotated[
        Confidence | None,
        Field(description="关系抽取模型或大模型给出的置信度"),
    ] = None

    evidence_start: Annotated[
        int | None,
        Field(
            ge=0,
            description="支持该关系的证据在当前text中的起始字符位置",
        ),
    ] = None

    evidence_end: Annotated[
        int | None,
        Field(
            ge=0,
            description="支持该关系的证据在当前text中的结束字符位置",
        ),
    ] = None

    extraction_source: Annotated[
        str,
        Field(description="关系生成来源，例如LLM、BERT_RE或HUMAN"),
    ] = "LLM"

    @model_validator(mode="after")
    def validate_evidence_span(self) -> "RelationMention":
        """检查关系证据区间是否合法。"""

        if (
            self.evidence_start is not None
            and self.evidence_end is not None
            and self.evidence_end <= self.evidence_start
        ):
            raise ValueError(
                "关系证据区间必须满足 evidence_end > evidence_start"
            )

        return self


# 中文注释：全项目最关键的跨层标注契约，连接模型推理、人审、数据集和图谱写入。
class CanonicalAnnotation(BaseModel):
    """
    实体识别和关系抽取共用的统一标注对象。

    BIO数据和OpenNRE数据均应由该对象转换生成。
    """

    annotation_id: AnnotationId
    case_id: CaseId
    doc_id: DocumentId
    text_id: TextId

    text: Annotated[
        str,
        Field(description="当前标注任务对应的完整原始文本"),
    ]

    entities: Annotated[
        list[EntityMention],
        Field(description="当前文本中识别或标注出的实体列表"),
    ] = Field(default_factory=list)

    relations: Annotated[
        list[RelationMention],
        Field(description="当前文本中识别或标注出的关系列表"),
    ] = Field(default_factory=list)

    annotation_source: Annotated[
        str,
        Field(description="标注来源，例如LLM、DEEP_MODEL或HUMAN"),
    ]

    schema_version: SchemaVersion

    status: Annotated[
        AnnotationStatus,
        Field(description="当前统一标注对象的处理状态"),
    ] = AnnotationStatus.GENERATED

    created_at: Annotated[
        datetime,
        Field(description="标注对象创建时间"),
    ] = Field(default_factory=datetime.now)

    updated_at: Annotated[
        datetime,
        Field(description="标注对象最近更新时间"),
    ] = Field(default_factory=datetime.now)

    metadata: Annotated[
        dict[str, Any],
        Field(description="提示词版本、模型名称等扩展信息"),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_annotation(self) -> "CanonicalAnnotation":
        """检查实体位置、关系引用和重复关系。"""

        entity_ids: set[str] = set()

        for entity in self.entities:
            if entity.entity_id in entity_ids:
                raise ValueError(
                    f"实体ID重复：{entity.entity_id}"
                )

            entity_ids.add(entity.entity_id)

            if entity.end > len(self.text):
                raise ValueError(
                    f"实体位置越界：{entity.name}，"
                    f"位置为[{entity.start}, {entity.end}]"
                )

            actual_text = self.text[entity.start:entity.end]

            if actual_text != entity.name:
                raise ValueError(
                    f"实体位置不匹配：期望'{entity.name}'，"
                    f"实际切片为'{actual_text}'"
                )

        relation_keys: set[
            tuple[str, str, RelationType]
        ] = set()

        for relation in self.relations:
            if relation.head_id not in entity_ids:
                raise ValueError(
                    f"关系头实体不存在：{relation.head_id}"
                )

            if relation.tail_id not in entity_ids:
                raise ValueError(
                    f"关系尾实体不存在：{relation.tail_id}"
                )

            relation_key = (
                relation.head_id,
                relation.tail_id,
                relation.type,
            )

            if relation_key in relation_keys:
                raise ValueError(
                    f"存在重复关系：{relation_key}"
                )

            relation_keys.add(relation_key)

            if (
                relation.evidence_end is not None
                and relation.evidence_end > len(self.text)
            ):
                raise ValueError(
                    f"关系证据位置越界：{relation.relation_id}"
                )

        return self


# =========================================================
# 5. 人工审核
# =========================================================

class ReviewRecord(BaseModel):
    """人工审核产生的记录。"""

    review_id: Annotated[
        str,
        Field(description="人工审核记录唯一编号"),
    ]

    annotation_id: AnnotationId

    reviewer_id: Annotated[
        str,
        Field(description="审核人员编号或账号"),
    ]

    decision: Annotated[
        ReviewDecision,
        Field(description="审核结论：通过、退回或拒绝"),
    ]

    comment: Annotated[
        str | None,
        Field(description="审核人员填写的修改说明"),
    ] = None

    modified_fields: Annotated[
        list[str],
        Field(description="人工修改过的字段名称列表"),
    ] = Field(default_factory=list)

    reviewed_at: Annotated[
        datetime,
        Field(description="人工审核完成时间"),
    ] = Field(default_factory=datetime.now)

    review_version: Annotated[
        int,
        Field(ge=1, description="当前标注对象的审核版本号"),
    ] = 1


# =========================================================
# 6. 数据集、训练与模型版本
# =========================================================

# 中文注释：记录一个不可变数据集版本的路径、schema、切分比例和样本统计。
class DatasetVersion(BaseModel):
    """一次被冻结并可复现的数据集版本。"""

    dataset_version: DatasetVersionName
    schema_version: SchemaVersion

    train_case_ids: Annotated[
        list[str],
        Field(description="训练集包含的案件ID"),
    ] = Field(default_factory=list)

    validation_case_ids: Annotated[
        list[str],
        Field(description="验证集包含的案件ID"),
    ] = Field(default_factory=list)

    test_case_ids: Annotated[
        list[str],
        Field(description="固定测试集包含的案件ID"),
    ] = Field(default_factory=list)

    train_size: Annotated[
        int,
        Field(ge=0, description="训练样本数量"),
    ] = 0

    validation_size: Annotated[
        int,
        Field(ge=0, description="验证样本数量"),
    ] = 0

    test_size: Annotated[
        int,
        Field(ge=0, description="测试样本数量"),
    ] = 0

    bio_train_uri: Annotated[
        str | None,
        Field(description="NER训练集BIO文件地址"),
    ] = None

    bio_validation_uri: Annotated[
        str | None,
        Field(description="NER验证集BIO文件地址"),
    ] = None

    bio_test_uri: Annotated[
        str | None,
        Field(description="NER测试集BIO文件地址"),
    ] = None

    re_train_uri: Annotated[
        str | None,
        Field(description="关系抽取训练集JSONL文件地址"),
    ] = None

    re_validation_uri: Annotated[
        str | None,
        Field(description="关系抽取验证集JSONL文件地址"),
    ] = None

    re_test_uri: Annotated[
        str | None,
        Field(description="关系抽取测试集JSONL文件地址"),
    ] = None

    status: Annotated[
        str,
        Field(description="数据集版本当前状态"),
    ] = "BUILDING"

    created_at: Annotated[
        datetime,
        Field(description="数据集版本创建时间"),
    ] = Field(default_factory=datetime.now)


class EvaluationMetrics(BaseModel):
    """NER或关系抽取模型的评价指标。"""

    precision: Annotated[
        float | None,
        Field(ge=0, le=1, description="精确率"),
    ] = None

    recall: Annotated[
        float | None,
        Field(ge=0, le=1, description="召回率"),
    ] = None

    f1: Annotated[
        float | None,
        Field(ge=0, le=1, description="F1值"),
    ] = None

    micro_precision: Annotated[
        float | None,
        Field(ge=0, le=1, description="Micro Precision"),
    ] = None

    micro_recall: Annotated[
        float | None,
        Field(ge=0, le=1, description="Micro Recall"),
    ] = None

    micro_f1: Annotated[
        float | None,
        Field(ge=0, le=1, description="Micro F1"),
    ] = None

    macro_precision: Annotated[
        float | None,
        Field(ge=0, le=1, description="Macro Precision"),
    ] = None

    macro_recall: Annotated[
        float | None,
        Field(ge=0, le=1, description="Macro Recall"),
    ] = None

    macro_f1: Annotated[
        float | None,
        Field(ge=0, le=1, description="Macro F1"),
    ] = None

    per_label_metrics: Annotated[
        dict[str, dict[str, float]],
        Field(description="各实体类别或关系类别的分项评价指标"),
    ] = Field(default_factory=dict)


# 中文注释：描述一次模型训练实验，用于关联数据集版本、超参数、指标和产物。
class TrainingRun(BaseModel):
    """一次完整且可复现的模型训练任务。"""

    training_run_id: Annotated[
        str,
        Field(description="训练任务唯一编号"),
    ]

    dataset_version: DatasetVersionName
    schema_version: SchemaVersion

    model_name: Annotated[
        str,
        Field(description="模型名称，例如 corruption_ner_bert_crf"),
    ]

    model_type: Annotated[
        str,
        Field(description="模型类型，例如 BERT_CRF、BERT_RE、CNN或PCNN"),
    ]

    status: Annotated[
        TaskStatus,
        Field(description="训练任务运行状态"),
    ] = TaskStatus.PENDING

    random_seed: Annotated[
        int,
        Field(description="训练所使用的随机种子"),
    ] = 42

    hyperparameters: Annotated[
        dict[str, Any],
        Field(description="批次、学习率、epoch等模型超参数"),
    ] = Field(default_factory=dict)

    checkpoint_uri: Annotated[
        str | None,
        Field(description="训练完成后模型权重的存储地址"),
    ] = None

    log_uri: Annotated[
        str | None,
        Field(description="训练日志文件地址"),
    ] = None

    git_commit: Annotated[
        str | None,
        Field(description="本次训练对应的Git代码提交编号"),
    ] = None

    metrics: Annotated[
        EvaluationMetrics | None,
        Field(description="本次训练获得的评价指标"),
    ] = None

    started_at: Annotated[
        datetime | None,
        Field(description="训练开始时间"),
    ] = None

    finished_at: Annotated[
        datetime | None,
        Field(description="训练结束时间"),
    ] = None

    error_message: Annotated[
        str | None,
        Field(description="训练失败时记录的错误信息"),
    ] = None


# 中文注释：描述可部署模型版本及其角色，为 Champion 模型选择和追溯提供依据。
class ModelVersion(BaseModel):
    """完成注册的模型版本。"""

    model_version: ModelVersionName

    training_run_id: Annotated[
        str,
        Field(description="产生当前模型版本的训练任务编号"),
    ]

    dataset_version: DatasetVersionName

    model_name: Annotated[
        str,
        Field(description="模型名称"),
    ]

    checkpoint_uri: Annotated[
        str,
        Field(description="模型权重文件存储地址"),
    ]

    role: Annotated[
        ModelRole,
        Field(description="模型当前角色：Champion、Challenger或Archived"),
    ] = ModelRole.CHALLENGER

    metrics: Annotated[
        EvaluationMetrics,
        Field(description="模型在固定测试集上的评价指标"),
    ]

    created_at: Annotated[
        datetime,
        Field(description="模型版本注册时间"),
    ] = Field(default_factory=datetime.now)


# =========================================================
# 7. Neo4j关系事实
# =========================================================

# 中文注释：图谱中的 Claim 业务对象，将关系事实、头尾实体和证据跨度绑定在一起。
class GraphClaim(BaseModel):
    """准备写入Neo4j的一条带来源和证据的关系事实。"""

    claim_id: Annotated[
        str,
        Field(description="图谱关系事实唯一编号"),
    ]

    case_id: CaseId
    doc_id: DocumentId
    text_id: TextId

    head_entity_id: Annotated[
        str,
        Field(description="关系头实体编号"),
    ]

    tail_entity_id: Annotated[
        str,
        Field(description="关系尾实体编号"),
    ]

    relation: Annotated[
        RelationType,
        Field(description="关系类型"),
    ]

    evidence_text: Annotated[
        str,
        Field(description="能够直接支持该关系的原文证据"),
    ]

    evidence_start: CharacterStart
    evidence_end: CharacterEnd

    source_url: Annotated[
        str | None,
        Field(description="关系证据所在原始材料的URL"),
    ] = None

    confidence: Annotated[
        Confidence | None,
        Field(description="关系预测置信度"),
    ] = None

    model_version: Annotated[
        str | None,
        Field(description="抽取该关系所使用的模型版本"),
    ] = None

    dataset_version: Annotated[
        str | None,
        Field(description="模型训练所对应的数据集版本"),
    ] = None

    status: Annotated[
        ClaimStatus,
        Field(description="该关系是模型预测还是人工确认"),
    ] = ClaimStatus.MODEL_PREDICTED

    created_at: Annotated[
        datetime,
        Field(description="关系事实创建时间"),
    ] = Field(default_factory=datetime.now)


# =========================================================
# 8. 工作流共享状态
# =========================================================

# 中文注释：预留的工作流持久化状态模型；当前主流程尚未用它实现断点续跑。
class WorkflowState(BaseModel):
    """协调Agent与各工作流节点共享的任务状态。"""

    task_id: Annotated[
        str,
        Field(description="端到端工作流任务唯一编号"),
    ]

    case_id: Annotated[
        str | None,
        Field(description="当前任务关联的案件编号"),
    ] = None

    doc_id: Annotated[
        str | None,
        Field(description="当前任务关联的文档编号"),
    ] = None

    annotation_id: Annotated[
        str | None,
        Field(description="当前任务关联的标注编号"),
    ] = None

    stage: Annotated[
        WorkflowStage,
        Field(description="当前任务所处的工作流阶段"),
    ]

    status: Annotated[
        TaskStatus,
        Field(description="当前任务运行状态"),
    ] = TaskStatus.PENDING

    input_uri: Annotated[
        str | None,
        Field(description="当前步骤输入文件或资源地址"),
    ] = None

    output_uri: Annotated[
        str | None,
        Field(description="当前步骤输出文件或资源地址"),
    ] = None

    schema_version: SchemaVersion

    model_version: Annotated[
        str | None,
        Field(description="当前步骤使用的模型版本"),
    ] = None

    dataset_version: Annotated[
        str | None,
        Field(description="当前步骤关联的数据集版本"),
    ] = None

    confidence: Annotated[
        Confidence | None,
        Field(description="当前任务结果的综合置信度"),
    ] = None

    review_required: Annotated[
        bool,
        Field(description="当前任务结果是否需要人工审核"),
    ] = False

    retry_count: Annotated[
        int,
        Field(ge=0, description="当前任务已经重试的次数"),
    ] = 0

    error_code: Annotated[
        str | None,
        Field(description="结构化错误代码"),
    ] = None

    error_message: Annotated[
        str | None,
        Field(description="具体错误说明"),
    ] = None

    context: Annotated[
        dict[str, Any],
        Field(description="当前任务运行过程中需要传递的少量结构化上下文"),
    ] = Field(default_factory=dict)

    created_at: Annotated[
        datetime,
        Field(description="任务创建时间"),
    ] = Field(default_factory=datetime.now)

    updated_at: Annotated[
        datetime,
        Field(description="任务最近更新时间"),
    ] = Field(default_factory=datetime.now)


# =========================================================
# 9. 深度学习模型统一输入输出
# =========================================================

class EntityPrediction(BaseModel):
    """BERT-CRF 输出的统一实体预测。"""

    entity_id: str
    name: str
    entity_type: Literal["PER", "ORG", "POSITION", "MONEY"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_prediction_span(self) -> "EntityPrediction":
        """保证实体采用合法的左闭右开字符区间。"""

        if self.end <= self.start:
            raise ValueError("实体预测必须满足 end > start")
        if not self.name:
            raise ValueError("实体预测名称不能为空")
        return self


class RelationPrediction(BaseModel):
    """BERTEntity 输出的统一关系预测。"""

    relation_id: str
    head_id: str
    tail_id: str
    relation_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_prediction_span(self) -> "RelationPrediction":
        """校验实体引用和可选证据区间的基础结构。"""

        if self.head_id == self.tail_id:
            raise ValueError("关系预测的 head_id 和 tail_id 不能相同")
        if (
            self.evidence_start is not None
            and self.evidence_end is not None
            and self.evidence_end <= self.evidence_start
        ):
            raise ValueError("关系证据必须满足 evidence_end > evidence_start")
        return self


# 中文注释：双模型推理的标准输出，汇总全文实体、关系、模型版本和诊断信息。
class ModelExtractionResult(BaseModel):
    """NER 到 RE 端到端推理的统一结果。"""

    text: str
    entities: list[EntityPrediction] = Field(default_factory=list)
    relations: list[RelationPrediction] = Field(default_factory=list)
    ner_model_version: str
    relation_model_version: str
    inference_seconds: float = Field(ge=0.0)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_extraction(self) -> "ModelExtractionResult":
        """校验实体原文切片及关系引用完整性。"""

        entity_ids: set[str] = set()
        for entity in self.entities:
            if entity.entity_id in entity_ids:
                raise ValueError(f"实体预测 ID 重复：{entity.entity_id}")
            entity_ids.add(entity.entity_id)
            if entity.end > len(self.text):
                raise ValueError(f"实体预测越界：{entity.entity_id}")
            if self.text[entity.start:entity.end] != entity.name:
                raise ValueError(f"实体预测位置不匹配：{entity.entity_id}")

        for relation in self.relations:
            if relation.head_id not in entity_ids:
                raise ValueError(f"关系头实体不存在：{relation.head_id}")
            if relation.tail_id not in entity_ids:
                raise ValueError(f"关系尾实体不存在：{relation.tail_id}")
        return self


class EvaluationResult(BaseModel):
    """统一模型评估结果。"""

    task_type: Literal["ner", "relation", "end_to_end"]
    model_version: str
    dataset_version: str
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# 中文注释：Trainer 对外返回的统一结果，供 Flow 汇总并生成模型 manifest。
class TrainingResult(BaseModel):
    """统一模型训练及实验清单结果。"""

    task_type: Literal["ner", "relation"]
    model_version: str
    dataset_version: str
    schema_version: str
    random_seed: int
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    checkpoint_path: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# 12. 文本处理与模型输入对象


# 中文注释：文本处理阶段的原始输入，可直接携带内容，也可引用受支持的本地文档。
class RawDocument(BaseModel):
    """保留来源信息的原始文档描述；模型本身不读取文件。"""

    doc_id: str = Field(min_length=1)
    doc_version_id: str | None = None
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    title: str | None = None
    published_at: datetime | None = None
    content_type: str | None = None
    raw_text: str | None = None
    local_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_sha256: str | None = None

    @model_validator(mode="after")
    def validate_content_source(self) -> "RawDocument":
        """保证内联正文和本地文件至少存在一种。"""

        if self.raw_text is None and not self.local_path:
            raise ValueError("raw_text 和 local_path 至少提供一个")
        return self


class ParseResult(BaseModel):
    """文档解析器的统一输出。"""

    text: str
    title: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str
    parser_version: str | None = None
    warnings: list[str] = Field(default_factory=list)


class CleanTextResult(BaseModel):
    """确定性清洗结果及双向字符映射。"""

    text: str
    original_to_cleaned_mapping: list[int] | None = None
    cleaned_to_original_mapping: list[int] | None = None
    warnings: list[str] = Field(default_factory=list)


class TextSegment(BaseModel):
    """清洗后全文中的段落或句子区间。"""

    segment_id: str
    segment_type: Literal["paragraph", "sentence"]
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    paragraph_id: str | None = None
    sentence_id: str | None = None
    order: int = Field(ge=0)


class TextWindow(BaseModel):
    """用于规则及 LLM 相关性判断的全文字符窗口。"""

    window_id: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    segment_ids: list[str] = Field(default_factory=list)
    rule_score: float = Field(default=0.0, ge=0, le=1)
    relevance_status: Literal["relevant", "irrelevant", "uncertain"] = (
        "uncertain"
    )
    relevance_type: str | None = None
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)
    llm_score: float | None = Field(default=None, ge=0, le=1)
    processing_source: Literal["rule", "llm", "combined"] = "rule"


# 中文注释：相关性过滤后保留下来的全文字符区间，是模型切块的直接上游。
class RelevantSpan(BaseModel):
    """合并后的连续相关证据范围。"""

    span_id: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    source_window_ids: list[str] = Field(default_factory=list)
    relevance_types: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=1)


# 中文注释：送入本地模型的最小文本块，保存全文 offset、token 数和是否可推理状态。
class ModelInputChunk(BaseModel):
    """可以直接传给 BERT-CRF predictor 的文本块。"""

    chunk_id: str
    case_id: str
    text_id: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    source_span_ids: list[str] = Field(default_factory=list)
    source_segment_ids: list[str] = Field(default_factory=list)
    token_count: int | None = Field(default=None, ge=0)
    overlap_left: int = Field(default=0, ge=0)
    overlap_right: int = Field(default=0, ge=0)
    model_ready: bool = True


class DeterministicFilterResult(BaseModel):
    """文档级确定性过滤结果。"""

    status: Literal["relevant", "irrelevant", "uncertain"]
    reason_codes: list[str] = Field(default_factory=list)


class RuleRelevanceResult(BaseModel):
    """单个相关性窗口的规则评分结果。"""

    score: float = Field(ge=0, le=1)
    matched_positive_terms: list[str] = Field(default_factory=list)
    matched_negative_terms: list[str] = Field(default_factory=list)
    status: Literal["relevant", "irrelevant", "uncertain"]
    reason_codes: list[str] = Field(default_factory=list)


class RelevanceEvidence(BaseModel):
    """LLM 返回的窗口内局部证据范围。"""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str | None = None


# 中文注释：LLM 相关性复核的结构化输出协议，避免上层直接依赖自由文本回答。
class RelevanceJudgment(BaseModel):
    """与现有 relevance_filter_prompt 输出兼容的结构化判断。"""

    relevant: bool
    score: float = Field(ge=0, le=1)
    relevance_types: list[str] = Field(default_factory=list)
    evidence_spans: list[RelevanceEvidence] = Field(default_factory=list)
    reason: str = ""


class ConflictDetail(BaseModel):
    """深度模型与 LLM 标注之间的单项差异说明。"""

    item_type: Literal["entity", "relation", "offset", "direction", "other"]
    item_id: str | None = None
    deep_model_result: dict[str, Any] | None = None
    llm_result: dict[str, Any] | None = None
    resolution: Literal[
        "USE_DEEP_MODEL", "USE_LLM", "MERGE", "HUMAN_REVIEW"
    ]
    reason: str


class ConflictReviewResult(BaseModel):
    """深度模型结果与 LLM 结果的结构化冲突复核输出。"""

    decision: Literal[
        "USE_DEEP_MODEL", "USE_LLM", "MERGE", "HUMAN_REVIEW"
    ]
    review_required: bool
    selected_entities: list[EntityMention] = Field(default_factory=list)
    selected_relations: list[RelationMention] = Field(default_factory=list)
    conflicts: list[ConflictDetail] = Field(default_factory=list)
    reason: str

    @model_validator(mode="after")
    def validate_review_decision(self) -> "ConflictReviewResult":
        """HUMAN_REVIEW 必须进入人工审核，自动结论则不得误标。"""

        expected = self.decision == "HUMAN_REVIEW"
        if self.review_required != expected:
            raise ValueError(
                "review_required 必须与 decision=HUMAN_REVIEW 保持一致"
            )
        entity_ids = {item.entity_id for item in self.selected_entities}
        if len(entity_ids) != len(self.selected_entities):
            raise ValueError("selected_entities 存在重复 entity_id")
        for relation in self.selected_relations:
            if (
                relation.head_id not in entity_ids
                or relation.tail_id not in entity_ids
            ):
                raise ValueError("selected_relations 引用了不存在的实体")
        return self


# 中文注释：单份原始文档完成解析、过滤和切块后的汇总结果，包含状态、错误和诊断信息。
class ProcessedCase(BaseModel):
    """从原始文档生成的可追溯文本处理结果。"""

    case_id: str
    doc_id: str
    doc_version_id: str | None = None
    title: str | None = None
    source_id: str
    source_url: str | None = None
    published_at: datetime | None = None
    original_text: str
    cleaned_text: str
    paragraphs: list[TextSegment] = Field(default_factory=list)
    sentences: list[TextSegment] = Field(default_factory=list)
    windows: list[TextWindow] = Field(default_factory=list)
    relevant_spans: list[RelevantSpan] = Field(default_factory=list)
    model_input_chunks: list[ModelInputChunk] = Field(default_factory=list)
    original_to_cleaned_mapping: list[int] | None = None
    cleaned_to_original_mapping: list[int] | None = None
    processing_status: Literal["ready", "irrelevant", "partial", "failed"]
    parser_name: str
    parser_version: str | None = None
    processing_errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def validate_text_slices(self) -> "ProcessedCase":
        """统一验证所有输出对象的清洗全文字符切片。"""

        items = [
            *self.paragraphs,
            *self.sentences,
            *self.windows,
            *self.relevant_spans,
            *self.model_input_chunks,
        ]
        for item in items:
            if item.end > len(self.cleaned_text):
                raise ValueError(f"文本范围越界：{item}")
            if self.cleaned_text[item.start:item.end] != item.text:
                raise ValueError(f"文本范围无法还原原文：{item}")
        return self


# 中文注释：总流程的跨阶段结果摘要，汇总标注、数据集、训练和图谱阶段输出。
class IngestionFlowResult(BaseModel):
    """Serializable summary returned by the top-level Prefect ingestion flow."""

    flow_run_id: str
    batch_id: str | None = None
    case_id: str | None = None
    workflow_phase: Literal[
        "automated",
        "human_review_submission",
        "human_review_consumption",
    ] = "automated"
    retrieval_result: dict[str, Any] | None = None
    annotation_result: dict[str, Any] | None = None
    review_sync_result: dict[str, Any] | None = None
    dataset_result: dict[str, Any] | None = None
    training_result: dict[str, Any] | None = None
    graph_ingestion_result: dict[str, Any] | None = None
    status: Literal[
        "completed", "completed_with_errors", "dry_run"
    ] = "completed"
    started_at: datetime
    finished_at: datetime
    errors: list[str] = Field(default_factory=list)

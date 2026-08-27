# 第二阶段：知识问答基础架构

本目录只读取第一阶段已经构建的 Neo4j 知识图谱，并把当前 Session 临时上传的文件作为另一类知识来源。它不创建永久知识库、不重新抽取实体关系，也不执行图谱写入。

## 数据来源与边界

- 上传文件：支持 TXT、PDF、DOCX；HTML 已提供基础实现，XLSX/PPTX 提供依赖可用时的基础实现。
- 图谱：通过 `FirstStageGraphAdapter` 调用第一阶段 `Neo4jService` 的只读能力。
- LLM：生产模式可通过 `FirstStageLLMClient` 复用第一阶段 `LLMService`，测试与演示使用 `MockLLMClient`。
- 临时文档仅保存在 `SessionDocumentStore` 内存中，进程结束即清除。

## 文档向量召回与重排

文档检索默认使用 `hybrid` 模式：BM25 和向量召回各自取得候选并集，再按 BM25 分数、向量余弦相似度和查询词覆盖率进行混合重排。每条文档 Evidence 的 `metadata.retrieval` 会返回 `bm25_score`、`vector_score`、`coverage_score`、`rerank_score` 和实际检索模式。

默认 `QA_EMBEDDING_PROVIDER=hashing`，使用本地确定性哈希向量，不下载模型、不调用网络，适合 Mock、开发和流程验证。Session Chunk 的向量缓存在当前 `SessionDocumentStore` 中，Session 删除或进程结束后随临时数据一起清除，因此没有写入永久向量库。

需要真实语义向量时，可以显式启用第一阶段 `.env` 中的 `EMBED_API_KEY`、`EMBED_MODEL_NAME` 和 `EMBED_MODEL_TYPE`：

```bash
QA_RETRIEVAL_MODE=hybrid \
QA_EMBEDDING_PROVIDER=first_stage \
QA_API_MODE=production \
第二阶段/.venv/bin/python -m uvicorn 第二阶段.api.app:app \
  --host 127.0.0.1 --port 8000
```

`first_stage` 模式会把上传 Chunk 和用户问题发送给已配置的 Embedding 服务，应在确认数据外发策略后启用。只需关键词检索时，可设置 `QA_RETRIEVAL_MODE=bm25`。候选数量、最低向量分数和三项重排权重均可通过 `.env.example` 中的 `QA_VECTOR_*`、`QA_RERANK_*` 参数调整。

## 调用链

```text
User Uploaded File -> ParserRouter -> Parser -> Chunker -> SessionDocumentStore
User Question -> QueryRouter -> DOCUMENT / GRAPH / HYBRID
-> DocumentRetriever / GraphRetriever -> EvidenceFusion
-> ContextBuilder -> PromptBuilder -> LLMClient -> AnswerResult
```

## Mock 模式运行

从项目根目录执行：

```bash
第一阶段/.venv/bin/python 第二阶段/main.py
```

如果已经激活第一阶段虚拟环境，也可以直接执行：

```bash
python 第二阶段/main.py
```

Mock 模式不连接 Neo4j、不调用网络 API，会直接输出完整 `AnswerResult` JSON。

## 测试

```bash
第二阶段/.venv/bin/python -m pytest -q 第二阶段/tests
```

## 模拟问答评测

内置 JSONL 模拟评测集覆盖 DOCUMENT、GRAPH、HYBRID、案件范围过滤和空问题校验。默认在进程内启动 Mock API，不连接 Neo4j 或外部 LLM：

```bash
第二阶段/.venv/bin/python -m 第二阶段.evaluation.runner \
  --output 第二阶段/evaluation/reports/mock-report.json
```

评测器会为每条样本创建独立 Session，按需上传模拟文档，然后检查 HTTP 状态码、路由、证据来源类型、实际检索模式、案件范围、最小证据数和回答是否为空。数据集位于 `evaluation/mock_qa_eval.jsonl`，可以逐行复制并替换为真实问题。

也可以评测已经启动的 API：

```bash
第二阶段/.venv/bin/python -m 第二阶段.evaluation.runner \
  --base-url http://127.0.0.1:8000 \
  --output 第二阶段/evaluation/reports/api-report.json
```

## FastAPI HTTP API

建议为第二阶段创建独立虚拟环境，避免修改已经冻结的第一阶段环境：

```bash
python3.13 -m venv 第二阶段/.venv
第二阶段/.venv/bin/python -m pip install -r 第二阶段/requirements.txt
```

Mock 模式启动，不连接真实 Neo4j 或 LLM：

```bash
QA_API_MODE=mock 第二阶段/.venv/bin/python -m uvicorn 第二阶段.api.app:app --host 127.0.0.1 --port 8000
```

正式模式复用第一阶段 Neo4j 和 LLM 配置：

```bash
QA_API_MODE=production 第二阶段/.venv/bin/python -m uvicorn 第二阶段.api.app:app --host 127.0.0.1 --port 8000
```

启动后可以访问：

- OpenAPI UI：`http://127.0.0.1:8000/docs`
- OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`
- Health：`http://127.0.0.1:8000/health`

第一版 Endpoint：

```text
GET    /health
POST   /sessions
POST   /sessions/{session_id}/documents
GET    /sessions/{session_id}/documents
POST   /sessions/{session_id}/questions
DELETE /sessions/{session_id}
```

上传文件只允许 `.txt/.pdf/.docx`，默认最大 20 MiB。Session、ParsedDocument 和 Chunk 均保存在内存中；上传文件只在解析期间写入临时文件，解析完成后立即删除。

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/sessions
curl -F 'file=@evidence.txt' http://127.0.0.1:8000/sessions/SESSION_ID/documents
curl -H 'Content-Type: application/json' -d '{"question":"上传材料中如何描述项目？"}' http://127.0.0.1:8000/sessions/SESSION_ID/questions
curl -H 'Content-Type: application/json' -d '{"question":"罗某与哪些人存在关系？","case_id":"CASE_ID"}' http://127.0.0.1:8000/sessions/SESSION_ID/questions
curl -H 'Content-Type: application/json' -d '{"question":"查找罗某与李某之间的相似路径","case_id":"CASE_ID","search_scope":"same_case"}' http://127.0.0.1:8000/sessions/SESSION_ID/questions
curl -X DELETE http://127.0.0.1:8000/sessions/SESSION_ID
```

API 配置集中在 `config.py`，环境变量示例见 `.env.example`。CORS 默认只允许本地 `localhost`/`127.0.0.1` 的 3000 和 5173 端口，没有使用生产环境不安全的通配符。

图谱问题可以在请求体中传入可选的 `case_id`，实体和 Claim 查询将限制在该案件内。未传入 `case_id` 且同名实体分布于多个案件时，接口返回 `409`，并在 `candidate_case_ids` 中列出候选案件，避免匿名人物证据跨案混入。

## 多路径提取与路径相似度检索

当问题同时命中至少两个图谱实体并包含“路径”“关系链”或“链路”时，`GraphRetriever` 会在第一阶段的 `Entity <- HEAD/TAIL - Claim - HEAD/TAIL -> Entity` 模型上提取多条简单路径。路径默认最多 3 个 Claim 跳，拒绝包含重复实体或 `REJECTED` Claim 的路径，并继续执行 `case_id` 案件隔离。

示例问题：

```text
张某与王某之间有哪些路径？
查找张某与王某之间的相似路径
比较张某和王某的关系链路
```

普通路径证据的 `metadata.kind` 为 `path`；相似候选为 `similar_path`。两者都会返回 `path_entities`、`path_claims`、`directions`、`relation_signature`、`entity_type_signature`、`claim_ids`、`case_ids` 和 `document_ids`。相似候选还会返回以下明细：

```json
{
  "similarity": {
    "score": 0.95,
    "relation_sequence_score": 1.0,
    "entity_type_sequence_score": 1.0,
    "relation_overlap_score": 1.0,
    "length_score": 0.5,
    "orientation": "forward"
  }
}
```

相似度不比较人物姓名，而是组合比较带方向的关系类型序列、实体类型序列、关系多重集合重合度和路径长度，因此可以发现不同人员之间结构相似的腐败关系链。默认阈值为 `0.55`，候选先由案件、端点类型和关系类型在 Neo4j 中收窄，再在 Python 层重排，避免无界枚举。

### 跨案相似检索范围

`search_scope` 是显式的相似路径候选范围，默认且省略时均为 `same_case`，系统不会自动升级到跨案检索：

- `same_case`：只从锚点路径所在案件中取候选。`case_id` 可省略，但同名实体跨案时会返回 `409` 要求明确案件。
- `selected_cases`：只从 `selected_case_ids` 列出的一个或多个案件中取候选；必须同时提供用于定位锚点路径的 `case_id`。
- `all_cases`：在全库已归档案件中取候选；必须提供锚点 `case_id`，并且必须由调用方在请求中显式设为 `all_cases`。前端还会要求用户二次确认。

`selected_case_ids` 仅能与 `selected_cases` 一起使用。所有范围都会拒绝横跨多个案件的单条候选路径；跨案只表示“从多个案件各自的完整路径中寻找相似模式”。

```json
{
  "question": "查找谢晚林与刘某之间的相似路径",
  "case_id": "case-anchor",
  "search_scope": "selected_cases",
  "selected_case_ids": ["case-002", "case-003"]
}
```

```json
{
  "question": "查找谢晚林与刘某之间的相似路径",
  "case_id": "case-anchor",
  "search_scope": "all_cases"
}
```

相关参数：

```text
QA_GRAPH_PATH_MAX_HOPS=3                # 允许 1..5
QA_GRAPH_PATH_CANDIDATE_LIMIT=100       # 每条锚点路径最多读取的候选数
QA_GRAPH_PATH_SIMILARITY_THRESHOLD=0.55 # 允许 0..1
```

实现思路参考了 GitHub 上的 [NetworkX simple paths](https://github.com/networkx/networkx/blob/main/networkx/algorithms/simple_paths.py)、[NetworkX graph similarity](https://github.com/networkx/networkx/blob/main/networkx/algorithms/similarity.py) 和 [Neo4j APOC path explorer](https://github.com/neo4j/apoc/blob/dev/core/src/main/java/apoc/path/PathExplorer.java)。当前实现没有新增 NetworkX 或 APOC 运行依赖，而是针对本项目的 Claim 中心图模型做了纯 Cypher 和轻量序列评分改造。

## 生产依赖组装

生产入口应依次构造：

```python
adapter = FirstStageGraphAdapter()
repository = GraphRepository(adapter)
graph_retriever = GraphRetriever(repository)
llm_client = FirstStageLLMClient()
```

随后将它们与 Parser、Chunker、DocumentRetriever 等依赖注入 `KnowledgeQAPipeline`。Neo4j 和 LLM 的密钥及连接参数仍由第一阶段 `.env` 与配置加载器提供，第二阶段不复制敏感配置。

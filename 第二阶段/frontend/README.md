# 腐败知识图谱研判台（前端）

这是第二阶段 FastAPI 服务的配套中文前端，提供：

- 自动创建和重置临时 Session；
- 上传 TXT、PDF、DOCX 并展示切片数量；
- 向文档、Neo4j 图谱或两者发起知识问答；
- 传入可选 `case_id`，避免同名实体跨案混入；
- 可视化展示路径实体、HEAD/TAIL 方向，并可点击 Claim 查看原始证据；
- 分开展示普通路径与相似路径，展示综合相似度及各评分项；
- 通过 `same_case` / `selected_cases` / `all_cases` 显式选择相似路径范围，全库跨案检索需二次确认。

## 来源与许可证

界面工程基于 [FlorentB974/graphrag](https://github.com/FlorentB974/graphrag) 的 Next.js 前端改造，选用提交 `c7c1ea998c29e3d1ae58e789a3ba99634929f133`。上游采用 MIT License，完整文本保存在 `LICENSE.upstream.md`。

本项目已将上游的接口、品牌和主要页面重构为当前第二阶段 API 协议。未改动的通用组件仍保留，便于后续继续扩展会话历史、文档详情等能力。

## 本地运行

先从项目根目录启动后端：

```bash
QA_API_MODE=mock 第二阶段/.venv/bin/python -m uvicorn 第二阶段.api.app:app \
  --host 127.0.0.1 --port 8000
```

再启动前端：

```bash
cd 第二阶段/frontend
npm install
npm run dev
```

浏览器访问 `http://localhost:3000`。后端 CORS 默认已允许该地址。

如果后端不在当前主机的 8000 端口：

```bash
cp .env.local.example .env.local
```

然后在 `.env.local` 设置：

```text
NEXT_PUBLIC_API_URL=http://your-api-host:8000
```

## 已对接接口

```text
GET    /health
POST   /sessions
DELETE /sessions/{session_id}
POST   /sessions/{session_id}/documents
GET    /sessions/{session_id}/documents
POST   /sessions/{session_id}/questions
```

## 验证

```bash
npm run build
```

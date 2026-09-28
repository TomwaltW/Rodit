# Rodit

给淘宝 / 抖店 / 拼多多商家用的 AI 客服：回答买家的售前、售后问题；查订单、查物流、处理售后是回答时调用的工具。
MVP 是**智能辅助**：AI 起草回答、给出依据和风险提示，客服确认后发送；售后操作只给建议，客服确认后才执行。

- 设计：`电商平台AI客服方案_v0.2.md`（下称「方案」）
- 代码结构、模块契约、分工：`docs/架构与分工.md`
- 开发约定（数据合规、云端开发硬规则、选型记录）：`CLAUDE.md`

## 快速开始

```bash
uv sync                          # 装依赖（Python ≥ 3.11）
uv run pytest                    # 全部测试（不联网；PostgreSQL 测试连不上时自动跳过）
uv run ruff check . && uv run ruff format --check .

bash scripts/dev-pg.sh           # 可选：启动本机 PostgreSQL，跑存储层测试
uv run python -m rodit.app.demo  # 用合成数据跑方案 §11 的核心流程（不需要任何密钥）
```

演示默认用离线替身：路由用关键词规则、生成只复述依据、质检用启发式判断，所以能在没有大模型的
环境里跑通"懂→找→写→判"全链路。设置 `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` / `RODIT_LLM_API_KEY`
后改用 OpenAI 兼容接口的境内已备案模型（只在本地开发机上这样做，见下方"数据合规"）。

## 生产组装需要的环境变量

`rodit.app.production.build_production_unit` 只从部署环境（平台云内）读这些变量；仓库、云端开发环境里都不放值：

| 变量 | 用途 |
|---|---|
| `RODIT_DATABASE_DSN` | PostgreSQL 连接串（应用以 `rodit_app` 角色访问，行级安全按店铺隔离） |
| `RODIT_DB_STATEMENT_TIMEOUT_MS` / `RODIT_DB_LOCK_TIMEOUT_MS` | 每个事务的语句超时 / 等锁超时（默认 2000 / 1000 毫秒），超时按知识库故障降级（§12.2） |
| `RODIT_TOKEN_KEY` | 店铺令牌加密密钥（应用内 AES-256-GCM，数据库只存密文） |
| `RODIT_BUYER_SALT` | 买家 ID 哈希成 `buyer_ref` 用的盐 |
| `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` / `RODIT_LLM_API_KEY` / `RODIT_LLM_TIMEOUT_S` | 平台云内境内已备案大模型（OpenAI 兼容接口） |

## 一条买家消息怎么走（方案 §4.2）

```
平台事件 → 接入适配器（平台差异止于此，买家 ID 哈希成 buyer_ref）→ 店铺授权检查
  → 问答引擎：去重 → 会话理解 → 路由（判断层 + §6.6 八条规则）→ 检索（知识中心 + 实时工具）
             → 生成（FAQ / 模板优先，大模型只按依据写）→ 质检（硬规则 + 判断层）
  → 推荐卡片（回答 + 依据 + 置信度 + 风险提示）→ 客服发送 / 编辑后发送 / 不合适 → 反馈闭环
```

售后审核是另一条链路：售后单 → R1–R8 规则建议 → 状态机 → 客服确认 → 重读平台状态 → 带幂等键调用写接口。

## 模块

| 包 | 内容 | 方案 |
|---|---|---|
| `rodit.core` | 契约：数据模型、Jev 格式判断层接口、端口协议、配置、证据规则、个人信息最小化 | §5–§10 |
| `rodit.decision` / `rodit.routing` | 判断层（规则、境内大模型投票、分类器、受限的 Jev）与路由 | §6 |
| `rodit.knowledge` | 知识中心：结构化优先检索、FAQ 语义检索、冲突处理、平台规则、问卷、缺失检查 | §5.2 |
| `rodit.tools` / `rodit.aftersale` | 平台只读工具（缓存降级）、物流问答与异常、售后问答与审核助手 | §7 |
| `rodit.generation` / `rodit.qa` | 答案生成与质检 | §5.4、§5.5 |
| `rodit.adapters` / `rodit.feedback` | 三平台接入（模拟）、模式 C、授权；反馈闭环与报表 | §8、§5.8、§12.4 |
| `rodit.engine` | 编排：会话理解、执行计划、推荐卡片、降级与兜底 | §5.3、§5.6、§5.7、§12.2 |
| `rodit.storage` / `rodit.llm` | PostgreSQL（行级安全）存储；OpenAI 兼容大模型客户端与用量计量 | §9.2、§10、§12.4 |
| `rodit.app` | 平台单元组装、应用服务、本地演示、生产组装 | §9.1 |
| `rodit.eval` | 合成评测集、§13.2 评分、§13.3 阈值校准 | §13 |

## 数据合规（摘自 CLAUDE.md）

- 买家和订单数据只在对应平台云内处理，不出境；本仓库只有**合成数据**（`rodit.testing.fixtures`）。
- 仓库和云端环境变量里没有任何生产密钥；密钥只从部署环境读（见 `rodit.app.production`）。
- 平台开放接口一律用模拟实现；真实实现要在平台云内开发和部署。
- 提示词、判断层请求、审计日志里不放买家昵称、手机号、收件地址（`rodit.core.pii.mask_pii`）。
- 售后写操作一律要客服确认（`HumanConfirmation`），执行前重读平台状态、带幂等键。

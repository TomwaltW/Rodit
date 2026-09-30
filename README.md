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
uv run python -m rodit.api       # 本地接口 + 模式 C 网页工具 http://127.0.0.1:8000/assist、商家控制台 /console
                                 # （启动时打印仅本地演示用的合成令牌，粘贴到网页登录）
uv run python -m rodit.eval      # §13 评测（324 条合成用例；--llm env 用真实大模型）
```

演示默认用离线替身：路由用关键词规则、生成只复述依据、质检用启发式判断，所以能在没有大模型的
环境里跑通"懂→找→写→判"全链路。设置 `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` / `RODIT_LLM_API_KEY`
后改用 OpenAI 兼容接口的境内已备案模型（只在本地开发机上这样做，见下方"数据合规"）。

### 用真实大模型评测

`python -m rodit.eval` 默认用离线替身跑合成评测集（方案 §13，不联网，是回归基线）。加 `--llm env` 改用真实的
OpenAI 兼容模型走完整链路：路由、质检按 `EngineConfig.decision` 组装（默认 `incloud_llm`：路由 3 次投票、
质检 1 次），生成走大模型，和演示单元接大模型时同一套接线。评测集全是合成数据，只在本地开发机上跑。

```bash
# 密钥只放本地 .env（已 gitignore）或云端环境的密钥设置；不要写进仓库、命令行历史或云端环境变量
set -a; source .env; set +a      # 提供 RODIT_LLM_BASE_URL / RODIT_LLM_MODEL / RODIT_LLM_API_KEY（可选 RODIT_LLM_TIMEOUT_S）

uv run python -m rodit.eval --llm env --limit 10 --seed 1          # 先随机抽 10 条试跑（省钱）
uv run python -m rodit.eval --llm env --intent size_advice         # 只跑一个问题类型
uv run python -m rodit.eval --llm env --json ~/rodit-evals/llm.json \
    --price-input 0.002 --price-output 0.008                       # 全量 + 完整结果 + 成本估算
uv run python -m rodit.eval --llm env --baseline ~/rodit-evals/llm.json --tolerance 0.02  # §13.5 回归
```

- 报告在离线评测的各节（§13.2 评分与上线门槛、§13.3 阈值校准）之外，还有 §12.1 耗时（各环节和端到端的
  P50 / P95，对照目标）、§14 用量（按路由 / 生成 / 质检分的 token、失败次数，给了单价时的成本，
  按 §14 示例口径折算的每店月用量）、降级统计；失败清单里有模型草稿和质检原因（已脱敏）。
- 单价是每 1000 token 的价格，也可以用环境变量 `RODIT_LLM_PRICE_INPUT_PER_1K` / `RODIT_LLM_PRICE_OUTPUT_PER_1K`；
  不给只报 token。`--config config/rodit.yaml` 可以换引擎配置（投票次数、超时、阈值）做对比。
- `--json out.json` 写完整结果（逐条的类型、评分、耗时、用量；放仓库外，不提交），`--json` 不带路径时输出到
  标准输出。
  `--baseline` 只比较两次都有的用例：类型准确率、路由、事实正确、合规满分率（全部和其中的测试集）任一项比基线低超过
  容差（默认 0，§13.5 指标不下降）时退出码为 1；配置错误（缺环境变量、连通性检查失败等）退出码为 2。
- 跑之前先发一个几个 token 的连通性检查请求，地址、模型、密钥不对时直接报错。报告和日志里没有密钥，
  接口地址只显示主机名。`--seed` 只决定 `--limit` 抽哪几条，不发给模型（路由投票要靠采样的随机性，§6.4）。

## 生产组装需要的环境变量

`rodit.app.production.build_production_unit` 只从部署环境（平台云内）读这些变量；仓库、云端开发环境里都不放值：

| 变量 | 用途 |
|---|---|
| `RODIT_DATABASE_DSN` | PostgreSQL 连接串（应用以 `rodit_app` 角色访问，行级安全按店铺隔离） |
| `RODIT_DB_STATEMENT_TIMEOUT_MS` / `RODIT_DB_LOCK_TIMEOUT_MS` | 每个事务的语句超时 / 等锁超时（默认 2000 / 1000 毫秒），超时按知识库故障降级（§12.2） |
| `RODIT_TOKEN_KEY` | 店铺令牌加密密钥（应用内 AES-256-GCM，数据库只存密文） |
| `RODIT_BUYER_SALT` | 买家 ID 哈希成 `buyer_ref` 用的盐 |
| `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` / `RODIT_LLM_API_KEY` / `RODIT_LLM_TIMEOUT_S` | 平台云内境内已备案大模型（OpenAI 兼容接口） |

## 部署：一个平台单元（方案 §9.1、§12）

每个平台云（聚石塔 / 抖店云 / 多多云）部署一份，镜像相同，平台由 `RODIT_PLATFORM` 决定（D1 未定，代码与平台无关）。
入口是 `rodit.ops`：`build_production_unit` + `create_app` + 运维接口 + JSON 日志。一个单元跑**一个进程**
（会话串行锁、适配器去重、短时缓存都在进程内；多副本要等换 Redis，见 CLAUDE.md 选型记录）。

### 构建与本地试用（只用合成数据）

```bash
docker build -t rodit:local .                  # 多阶段：uv sync --frozen --no-dev，非 root（uid 10001）
cp deploy/env.example deploy/.env              # 填随机值（openssl rand -hex 32）；deploy/.env 不进仓库
docker compose -f deploy/docker-compose.yml up --build -d
docker compose -f deploy/docker-compose.yml logs init   # 试用令牌只在第一次初始化时打印
curl -s http://127.0.0.1:8000/readyz
curl -s -X POST -H 'Authorization: Bearer <客服令牌>' -H 'Content-Type: application/json' \
  -d '{"session_id":"web_1","question":"能开发票吗"}' http://127.0.0.1:8000/v1/shops/s_1001/standalone/ask
docker compose -f deploy/docker-compose.yml down -v    # 删掉全部试用数据和令牌
```

compose 里三个服务：`db`（PostgreSQL 16；应用账号 `rodit_login` 非超级用户、属于 `rodit_app`，行级安全生效）、
`init`（`local-init`：迁移 → 写入合成演示数据 → 生成令牌文件，只存哈希；可以重复执行）、
`app`（`RODIT_ENV=local`：合成平台数据 + 模拟适配器，`deploy/.env` 没配大模型时用离线替身）。
平台云拉不到 Docker Hub 时用 `--build-arg PYTHON_IMAGE=… --build-arg UV_IMAGE=…` 换内网镜像。

### 命令（`python -m rodit.ops …`，镜像的 ENTRYPOINT）

| 命令 | 作用 |
|---|---|
| `serve` | 启动服务（镜像默认命令） |
| `migrate` | 迁移数据库（属主账号 `RODIT_MIGRATE_DSN`，没设就用 `RODIT_DATABASE_DSN`），并授权 `rodit_app` 读迁移版本（`/readyz` 用）；幂等，每次发布前跑一次 |
| `issue-token --shop-id … --user-id … --role agent\|manager\|merchant\|integration\|reviewer [--expires-days N] [--append 文件]` | 签发访问令牌：原文只打印这一次，令牌文件只存 SHA-256 |
| `issue-token --metrics --name prometheus [--append 文件]` | 签发 `/metrics` 的抓取令牌 |
| `local-init` | 只用于本地试用（`RODIT_ENV=local`） |
| `healthcheck` | 镜像 HEALTHCHECK 用：请求本机 `/healthz` |
| `recalibrate [--since …] [--until …] [--min-support N] [--out 文件] [--config YAML]` | §13.5 每月按采纳反馈重算阈值：只输出建议（中文摘要 + 只含改动的 YAML 片段），不改任何配置 |
| `train-classifier --out 文件 [--min-samples N] [--force] [--overwrite] [--config YAML]` | §6.4 用反馈标注训练路由分类器：报告留出集和评测集上对比关键词规则的效果，原子写入模型文件（JSON） |
| `onboard-shop --shop-id … (--token-stdin \| --token-file 文件) …` / `refresh-tokens` / `purge-expired [--dry-run]` / `revoke --shop-id …` / `set-status --shop-id … --status …` | §8.4 店铺生命周期（见下文"上线就绪"）。除了 `--dry-run` 都要拿单副本租约：`serve` 在跑时拒绝执行（退出码 2），日志写明改用哪个接口或 `serve` 里的哪个定时任务 |
| `record-eval --report 结果.json` / `acceptance-report --since … [--json]` / `compliance-scan [--week …]` | §13.2、§13.4、§15 上线门槛的数据和 Phase 1 验收报告 |

`recalibrate` 和 `train-classifier` 是离线任务，标注从 `RODIT_TRAINER_DSN` 读（这个登录账号要能
`SET ROLE rodit_trainer`，只能跨店读 `labeled_sample`，读不到别的表）；退出码 0 完成、1 数据库不可用、
2 参数或配置不对。两者的结果上线前都要先跑回归（§13.5）：
`python -m rodit.eval --config <新配置> [--classifier <模型文件>] --baseline <基线.json>`，指标不下降才上线。

### 环境变量

密钥只由平台云的密钥管理注入（环境变量或挂载文件），不进仓库、不进镜像、不进云端开发环境：

| 变量 | 用途 |
|---|---|
| `RODIT_DATABASE_DSN` | 服务用的连接串：登录角色属于 `rodit_app`，不能是超级用户、不能带 BYPASSRLS（§9.2） |
| `RODIT_MIGRATE_DSN` | 迁移用的属主账号，只给 `migrate` 任务，不给服务进程 |
| `RODIT_TOKEN_KEY` | 店铺令牌加密密钥 |
| `RODIT_BUYER_SALT` | `buyer_ref` 的盐（换了盐，已有会话就对不上了） |
| `RODIT_LLM_API_KEY` | 大模型密钥（缺失时 `/readyz` 报 `missing_api_key`，进程不接流量） |
| 令牌文件 `RODIT_TOKENS_FILE` | 默认 `/run/secrets/rodit_tokens.json`；按间隔（`RODIT_TOKENS_RELOAD_S`，默认 30 秒）检查变化，轮换不用重启 |

其余设置：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RODIT_ENV` | `production` | `local` 只用于本地试用（允许合成数据、离线大模型替身） |
| `RODIT_PLATFORM` | 必填 | `taobao` / `douyin` / `pdd` |
| `RODIT_PLATFORM_BINDINGS` | | 平台云代码的绑定工厂 `包.模块:函数`（见下） |
| `RODIT_MODE_C_ONLY` | `false` | 只部署模式 C（Phase 0，§8.1）：不需要平台绑定 |
| `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` / `RODIT_LLM_TIMEOUT_S` | | 境内已备案大模型（也可以写在配置文件 `llm` 段） |
| `RODIT_DB_STATEMENT_TIMEOUT_MS` / `RODIT_DB_LOCK_TIMEOUT_MS` | 2000 / 1000 | 请求事务的语句 / 等锁超时（§12.1、§12.2） |
| `RODIT_CONFIG` | 镜像里 `/app/config/rodit.yaml` | 引擎配置 |
| `RODIT_METRICS_PUBLIC` | `false` | `/metrics` 不要令牌（只在网络策略已经限制访问时用） |
| `RODIT_HOST` / `RODIT_PORT` | 镜像里 `0.0.0.0` / `8000` | 监听地址 |
| `RODIT_LOG_LEVEL` / `RODIT_LOG_FORMAT` | `INFO` / `json` | 日志 |
| `RODIT_MIGRATE_ON_START` | `false` | `serve` 前先迁移（服务进程需要属主权限，不推荐） |
| `RODIT_AUTH_EXPIRY_WARN_DAYS` / `RODIT_AUTH_EXPIRY_CHECK_S` | 7 / 300 | 授权即将过期告警的窗口、检查间隔（`serve` 的后台线程；0 = 不检查） |
| `RODIT_READINESS_CACHE_S` | 2 | `/readyz` 结果缓存 |
| `RODIT_READINESS_REFRESH_S` | 10 | 后台刷新就绪检查的间隔（0 = 只由 `/readyz` 刷新）；`/metrics` 只输出上次的结果 |
| `RODIT_TOKEN_REFRESH_INTERVAL_S` / `RODIT_PURGE_EXPIRED_INTERVAL_S` | 3600 / 21600 | `serve` 里令牌续期、过期清理的间隔（0 = 不跑），见"上线就绪" |
| `RODIT_CLASSIFIER_PATH` | | `decision.provider: classifier` 时必填：`train-classifier` 输出的模型文件；缺失或校验不过拒绝启动 |
| `RODIT_TRAINER_DSN` | | 只给离线任务（`recalibrate`、`train-classifier`）：能 `SET ROLE rodit_trainer` 的登录账号，不给服务进程 |

### 生产模式下拒绝启动（退出码 2，日志只写原因和变量名）

- 既没有平台绑定（`RODIT_PLATFORM_BINDINGS`），也没有明确 `RODIT_MODE_C_ONLY=1`；两个同时设置也不行；
- 平台绑定里有本仓库的模拟实现（`rodit.testing`、`rodit.tools.mock_platform`、各平台的 `*MockAdapter`）；
- 缺少 `RODIT_DATABASE_DSN`、`RODIT_TOKEN_KEY`、`RODIT_BUYER_SALT` 或大模型地址 / 模型名；
- 令牌文件不存在或格式不对；`/metrics` 既没有抓取令牌也没有 `RODIT_METRICS_PUBLIC=1`。

### 平台云代码要提供什么

- **平台绑定工厂**：`RODIT_PLATFORM_BINDINGS=包.模块:函数`，签名 `(rodit.ops.BindingContext) -> rodit.ops.PlatformBindings`。
  `tools` 实现 `ToolGateway`（只读，§7.1；限流 / 故障抛 `ToolUnavailable`，返回的数据已最小化）；
  `adapters` 实现 `PlatformAdapter`（模式 A / B；用 `ctx.hasher` 把平台买家 ID 变成 `buyer_ref`；`peek_shop_id` 不改状态）；
  `action_port` 二期才有（写售后，必须人确认）；`shop_ids` 返回订购列表（授权过期告警用）；
  `catalog` 实现 `ItemCatalog`（平台商品接口，只读）、`ocr` 实现 `ImageTextExtractor`（详情图 OCR，图片不出云），
  用于商家控制台的商品同步（§5.2.3），不给时同步接口返回 503。
- **令牌文件**：由平台的登录体系（工作台子账号等）签发令牌，写进挂载的密钥文件（格式见 `rodit.ops.tokens`）。
- **数据库账号**：属主账号给 `migrate` 任务；服务账号 `GRANT rodit_app TO <账号>`；
  离线训练 / 重算任务单独一个账号 `GRANT rodit_trainer TO <账号>`（`RODIT_TRAINER_DSN`）。
- 大模型：平台云内可调用的境内已备案模型（D5），OpenAI 兼容接口。

### 探针与指标（§12.4）

- `/healthz`：存活，不查依赖；
- `/readyz`：就绪 = 数据库能用应用角色连上 + 迁移到最新（库里有更新版本的迁移也算就绪）+ 大模型已配置（有密钥，不真调用）；
  503 时 `checks` 里有原因代码（`unreachable` / `timeout` / `not_migrated` / `behind` / `no_privilege` / `missing_api_key`…），没有任何连接串或店铺数据；
- `/metrics`（Prometheus 文本格式，要抓取令牌）：抓取时不查库（数据库网络不通时一次检查要十几秒，会超过抓取超时），
  就绪结果和授权过期检查由 `serve` 的后台线程刷新，输出上次的结果和 `rodit_readiness_age_seconds`（太旧按不就绪报）；
  `rodit_engine_latency_seconds{stage}`（端到端 `total` 和各环节，
  P50 / P95 用 `histogram_quantile`）、`rodit_engine_messages_total{shop_id,outcome}`、`rodit_suggestions_total`、
  `rodit_qa_checks_total` / `rodit_qa_failures_total`、`rodit_qa_hard_rule_hits_total{rule,blocking}`、
  `rodit_degradations_total{kind}`、`rodit_llm_tokens_total{shop_id,purpose,direction}`、
  `rodit_platform_tool_calls_total{tool,outcome}`、`rodit_auth_expiring_shops`、
  `rodit_http_errors_total{route,code}`（`code="tenant_isolation"` 即串店拦截）等。
  标签里只有店铺 ID 和代码，不含买家文本、`buyer_ref`、会话 ID；取消订购后该店铺的指标一起删掉。
- 日志：一行一个 JSON，手机号、地址等个人信息打码，令牌、连接串密码、密钥原文替换成 `***`。

## 商家控制台与上线节奏（方案 §9.1、§5.2、§13.4、§13.5）

店铺范围的接口都在 `/v1/shops/{shop_id}/…`，鉴权和串店检查同其他接口；操作人一律取自令牌身份。

| 功能 | 接口 | 角色 |
|---|---|---|
| FAQ：列表、录入、修改（旧条目下线、新条目上线，旧推荐的依据 ID 不变）、带修改地审核候选 | `console/faqs`、`console/faqs/{id}/edit`、`console/candidates/{id}/approve-edited` | 看：店长、商家；改：商家 |
| 店铺政策：逐项查看和修改（改动即时生效，§5.2.1） | `console/policies`、`console/policies/{key}` | 同上 |
| 商品卡片：商家填的字段（尺码建议、洗护、使用说明、不能说的卖点）；平台商品同步 + 详情图 OCR | `console/items`、`console/items/{item_id}`、`console/items/sync` | 同上 |
| 店铺配置（§9.3）：语气、称呼、长度、禁用词、关闭的问题类型、推荐条数、售后金额上限、店长 | `console/config` | 同上；影子模式、全自动开关不在这里 |
| 上线状态：当前阶段、影子对比、各阶段门槛、没法自动检查的条件 | `rollout/status`、`rollout/shadow` | 店长、商家、复核人员 |
| 切换影子 / 智能辅助（§13.4）：切到智能辅助要过门槛，没过要强制并写明原因 | `rollout/stage` | 切回影子：商家、复核人员；切到智能辅助：复核人员 |
| 每周人工复核（§13.5）：按问题类型分层抽 50 条、按 §13.2 四个维度打分、报表（事实正确和合规满分率 ≥ 95%） | `quality/sample`、`quality/reviews`、`quality/reviews/{id}`、`quality/report` | 复核人员；报表店长、商家也能看 |

网页控制台 `/console`（和 `/assist` 同样的技术和安全约束：原生 HTML / JS / CSS、严格 CSP、令牌只放请求头和
sessionStorage）按角色显示知识、政策、商品、设置、上线、复核、报表页签；两个页面共用登录状态，互相有链接。
复核人员（`reviewer` 角色）是我们的质检人员，按店铺签发令牌；看推荐原文每次都记访问审计（§9.2）。
写操作都和取消订购互斥；店铺配置和商品卡片的修改在仓储里原子地只改涉及的字段，并发修改不会互相覆盖。

## 上线就绪：店铺生命周期、5 秒预算、上线验收、运维加固（方案 §8.4、§12、§13、§15 Phase 1）

**店铺生命周期（§8.4）**：
- 开通：平台订购 / 授权推送 → `POST /v1/platform/{access}/subscription`（平台对接身份，只能推自己店铺的），
  或平台云代码进程内调用 `BindingContext.on_subscription`，或运维命令 `onboard-shop`（令牌只从标准输入或文件读，
  不进命令行参数）。开通时权限按 `tenancy.onboarding_phase` 核对（1 = MVP 只读；写售后要到二期）。
- 续期：给 `tenancy.refresh_ahead_hours` 内到期、有续期令牌的店铺续期；续不了的标记"要重新授权"，
  之后的处理一律拒绝。续期令牌同样应用内加密（附加认证数据和访问令牌不同）。
- 取消订购 / 授权过期：取消推送立即删除该店铺在本单元的数据；过期超过 `tenancy.purge_after_expiry_days`
  （默认 30 天，【待定：委托处理协议】）的按约定删除。删除后留一条不带店铺数据的系统事件
  `shop.purged`（只有店铺 ID、原因和各类数据的条数）。令牌文件里这家店的访问令牌要另外删掉（清理过、没有再开通的
  店铺不再计入授权过期告警）。`revoke --shop-id` 手动取消，`set-status` 切换试用 / 正式 / 停用。
- 续期和过期清理在 `serve` 进程里按间隔跑（`RODIT_TOKEN_REFRESH_INTERVAL_S` 默认 1 小时、
  `RODIT_PURGE_EXPIRED_INTERVAL_S` 默认 6 小时，起来 1 分钟后跑第一轮），不用另外排定时任务：取消订购的闸门
  （等在途处理做完再删）、工具缓存、适配器去重记录都在 `serve` 进程内，另起一个进程删数据时 `serve` 手上的消息
  照样把推荐和审计写进来、再也没有人删。没删完的店铺下一轮重试；出错只记日志和
  `rodit_background_task_runs_total{task,outcome}`（告警 `RoditBackgroundTaskFailing`）。
- 命令行里改数据的店铺生命周期命令（`onboard-shop`、`refresh-tokens`、`purge-expired`、`revoke`、`set-status`）
  和 `serve` 拿同一个单副本租约：`serve` 在跑时拒绝执行（退出码 2），改用 `POST /v1/platform/{access}/subscription`
  （开通 / 续订 / 取消订购）、商家令牌 `POST /v1/shops/{shop_id}/revoke`，或者 `serve` 里的定时任务；
  `set-status` 没有对应接口，停服务窗口里跑。`purge-expired --dry-run` 只读，可以和 `serve` 同时跑。
- 售后审核（拉单、协商、确认）和商品同步也先检查授权；确认执行同意退款 / 退货 / 拒绝还要有 `refund.write`。

**5 秒预算（§12.1）**：每条消息一个截止时间（`latency.message_budget_s`，默认 4.5 秒）；多问题分条、两种话术、
每条草稿的质检都并行跑，结果顺序和串行一样；剩余时间少于 `latency.min_llm_remaining_s` 就不再调大模型，退回
FAQ / 模板；超时的只留已经通过质检的 FAQ / 模板卡片，降级记 `budget`，截止之后不会再写任何记录。
耗时另记从收到消息算起的 `since_received`。"显示签收但没收到"有确定的模板：签收详情 + 驿站 / 快递柜取件指引 +
"还找不到可以在订单页申请售后"（§11.3）。

**多问题分条（§6.6 规则 5）**：投票制大模型判断层每票在同一个请求里多答一个追问"第二个问题类型"，规则 5 按
"被问到"的占比分条；站外交易即使只出现在追问里也按规则 1 给平台规则标准话术。

**上线门槛的数据（§13.2、§13.4、§15）**：
- 复核人员登记平台判定的违规和投诉：`POST /v1/shops/{shop_id}/rollout/incidents`（按周）；
- `compliance-scan [--week …]`：扫描实际发出去的文字（编辑后发送、客服和机器人的回复）里的站外引流、绝对化用语、
  店铺禁用词，只输出条数和 ID，不输出原文；
- `python -m rodit.eval --json 结果.json` 带配置指纹；`record-eval --report 结果.json` 把结论记下来，
  指纹和运行中的配置一致时才算测试集门槛的依据（离线替身的结论不算）；
- `acceptance-report --since … [--until …] [--json]`：按店铺和全单元汇总 Phase 1 验收（§13.2 门槛、连续两周
  采纳率 ≥ 60%、P95 ≤ 5 秒、违规为 0、每周复核事实正确和合规满分率 ≥ 95%），全部通过退出码 0，否则 1 并写明哪一项；
- 采纳率不再把全自动发出的回复算进去，单列"全自动"。

**运维加固（§9.1、§12.3、§12.4）**：
- 单副本租约：`serve` 启动时在 PostgreSQL 上拿按平台的会话级咨询锁，已有进程持有时拒绝启动（退出码 2，日志写明
  持有者）；`/readyz` 多一项 `lease`。会话锁、取消订购的闸门都在进程内，Phase 1 一个平台单元只跑一个副本；
  改数据的运维命令也拿这个租约（见上），运维命令在跑时 `serve` 起不来（退出码 2，命令做完后编排系统重启即可）；
  连接串要直连或走会话模式的连接池（事务模式的连接池拿不住会话锁）。
- 大模型出口可注入：平台绑定的 `llm_transport`（和 OpenAI 兼容客户端的 transport 同签名）给奇门 / EGW 等报备通道用，
  计量和就绪检查照旧。
- 模型公示：配置 `llm.public_name` / `llm.filing_no`（【待定：D5】），`/v1/me` 返回、`/assist` 页脚显示；
  `llm.require_disclosure: true` 时生产部署缺了拒绝启动。
- 指标：`rodit_tenant_isolation_total{stage}`（串店拦截：`http` 是调用方令牌不是这家店的，`server` 是接口处理中
  服务端发现的，其余是引擎在哪一步拦下的）、平台工具按 `unavailable / throttled / timeout` 分开、
  工具缓存命中 / 过期兜底、`rodit_ready` 和各项就绪检查、后台任务每一轮的结果；告警和预计算规则在 `deploy/prometheus/`
  （授权即将过期 / 已过期、任何串店、端到端 P95 > 5 秒、工具和大模型失败率、降级突增、就绪失败、抓不到指标、
  后台任务出错）。Prometheus 抓取任务的 `job_name` 要叫 `rodit`（`RoditScrapeDown` 按它找目标）。

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
| `rodit.knowledge` | 知识中心：结构化优先检索、FAQ 语义检索、冲突处理、平台规则、问卷、缺失检查；知识编辑、商品同步 | §5.2 |
| `rodit.tools` / `rodit.aftersale` | 平台只读工具（缓存降级）、物流问答与异常、售后问答与审核助手 | §7 |
| `rodit.generation` / `rodit.qa` | 答案生成与质检 | §5.4、§5.5 |
| `rodit.adapters` / `rodit.feedback` | 三平台接入（模拟）、模式 C、授权；反馈闭环与报表；影子对比、上线门槛、每周复核 | §8、§5.8、§12.4、§13.4、§13.5 |
| `rodit.engine` | 编排：会话理解、执行计划、推荐卡片、降级与兜底 | §5.3、§5.6、§5.7、§12.2 |
| `rodit.storage` / `rodit.llm` | PostgreSQL（行级安全）存储；OpenAI 兼容大模型客户端与用量计量 | §9.2、§10、§12.4 |
| `rodit.app` | 平台单元组装、应用服务、本地演示、生产组装 | §9.1 |
| `rodit.api` | HTTP 接口（按店铺鉴权、串店拦截）；商家控制台、上线节奏与复核接口；模式 C 独立话术助手网页 `/assist`、商家 / 复核控制台网页 `/console` | §8.1、§8.4、§9.1、§9.2、§13.4 |
| `rodit.ops` | 部署入口（`python -m rodit.ops`）、健康 / 就绪检查、Prometheus 指标、个人信息过滤的 JSON 日志 | §9.1、§12 |
| `rodit.eval` | 合成评测集（324 条）、§13.2 评分、§13.3 阈值校准；离线替身或真实大模型、耗时与成本、回归比较；每月阈值重算、路由分类器训练 | §13、§12.1、§14、§6.4 |

## 数据合规（摘自 CLAUDE.md）

- 买家和订单数据只在对应平台云内处理，不出境；本仓库只有**合成数据**（`rodit.testing.fixtures`）。
- 仓库和云端环境变量里没有任何生产密钥；密钥只从部署环境读（见 `rodit.app.production`）。
- 平台开放接口一律用模拟实现；真实实现要在平台云内开发和部署。
- 提示词、判断层请求、审计日志里不放买家昵称、手机号、收件地址（`rodit.core.pii.mask_pii`）。
- 售后写操作一律要客服确认（`HumanConfirmation`），执行前重读平台状态、带幂等键。

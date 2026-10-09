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

bash scripts/dev-pg.sh           # 可选：启动本机 PostgreSQL，跑存储层测试（macOS 上是 Docker 里的一次性实例，
                                 # 按它打印的那一行 export RODIT_TEST_PG_DSN=… 之后再跑 pytest）
uv run python -m rodit.app.demo  # 用合成数据跑方案 §11 的核心流程（不需要任何密钥）
uv run python -m rodit.showcase  # 客户演示站 http://127.0.0.1:8100/（不用登录，见下一节）
uv run python -m rodit.api       # 本地接口 + 模式 C 网页工具 http://127.0.0.1:8000/assist、商家控制台 /console、
                                 # 售后审核工作台 /aftersale（启动时打印仅本地演示用的合成令牌，粘贴到网页登录）
uv run python -m rodit.eval      # §13 评测（324 条合成用例；--llm env 用真实大模型）
uv run python -m rodit.eval.compare  # 判断层离线对比（§6.3、§6.4、§13.3；--jev env 跑 Jev，只发合成数据）
```

### 客户演示站（给商家演示用）

```bash
uv run python -m rodit.showcase                      # 打开 http://127.0.0.1:8100/
uv run python -m rodit.showcase --platform douyin    # 打开时先显示抖店（taobao / douyin / pdd）
uv run python -m rodit.showcase --host 0.0.0.0       # 局域网投屏（没有登录，会提醒；只有合成数据）
```

独立的 FastAPI 应用（`rodit.showcase`），不挂进生产的 `create_app`，平台单元的部署入口（`rodit.ops`）也不带它。
页面上跑的是真实的问答引擎、售后审核助手和报表，数据全是合成的、大模型是离线替身，不联网：

- **产品概览**：懂→找→写→判→兜底五步、决定架构的三个发现（§0）、§1.3 目标和演示单元的当前值、三平台对比（§2.1）。
- **智能辅助工作台**：预设的买家会话（§11 尺码咨询、物流查询、签收未收到、站外交易、质检拦截、多问题分条、
  情绪激动、投诉、价格、破损照片、尺码截图……），讲解的人也可以自己扮买家打字、发合成图片；右侧是推荐卡片
  （依据、置信度、风险提示，发送 / 编辑后发送 / 不合适），下面的「AI 是怎么想的」逐步展示路由概率、
  命中的 §6.6 规则、证据、质检分数和各环节耗时。半自动开放的类型标「已填入输入框」。拼多多是模式 C（§8.1）。
- **售后审核助手**：待审核售后单的 R1–R8 建议、§7.3.2 状态机、人确认后执行（重读状态、幂等键）。
- **数据看板**：采纳率趋势、问题类型分布、弃用原因、质检拦截、耗时，平台客服指标接入前后对比（§12.4）。
- **合规与部署**：每个平台云一个单元（§9.1）、个人信息打码的现场演示（§5.3）、话术硬规则、Jev 接口兼容。

三个平台各一个演示单元（§9.1），第一次打开某个平台时回放近 5 周的合成历史（约 1.5 秒），「重置演示」重新生成。
`--llm env` 用 `RODIT_LLM_*` 的真实大模型写回答（只允许本机地址；路由、质检、历史回放仍用离线替身），
只在本地开发机上用。§11.5 质检拦截场景的第一稿是演示脚本写的（页面上有标注），拦截和重写是真实的质检和引擎。

- 只认监听地址对应的 Host（防 DNS 重绑定）：默认用 http://127.0.0.1:8100/、http://localhost:8100/ 或
  http://[::1]:8100/ 打开；`--host` 指定了地址时只能用这个地址；`--host 0.0.0.0` / `::` 不限 Host。
  写操作的 Origin 和 Host 不一致（别的网站发来的跨站请求）返回 403。
- 写操作（推荐反馈、售后确认、协商）带页面载入时演示单元的编号：别的窗口重置过或服务重启过时返回
  409 `stale_epoch`，页面提示并重新载入，重置前的卡片不会落到重置后编号相同的推荐上。
- `--llm env` 时 `RODIT_LLM_*` 没设或格式不对就拒绝启动（退出码 2，提示里没有密钥）；「推荐出现时间」只算
  演示时现发的消息（回放的历史是离线替身，不计入），还没有时显示「暂无真实大模型数据」。
- 「平台话术违规」和上线门槛的合规扫描（`rodit.feedback.compliance`）用同一份词表，客服编辑时加的手机号、
  邮箱也算；「售后误操作」只展示写操作次数和防护措施，标「后续评估」（对错要人工复核）。

演示默认用离线替身：路由用关键词规则、生成只复述依据、质检用启发式判断，所以能在没有大模型的
环境里跑通"懂→找→写→判"全链路。设置 `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` / `RODIT_LLM_API_KEY`
后改用 OpenAI 兼容接口的境内已备案模型（只在本地开发机上这样做，见下方"数据合规"）。

### 用真实大模型评测

`python -m rodit.eval` 默认用离线替身跑合成评测集（方案 §13，不联网，是回归基线）。加 `--llm env` 改用真实的
OpenAI 兼容模型走完整链路：路由、质检按 `EngineConfig.decision` 组装（默认 `incloud_llm`：路由 3 次投票、
质检 1 次），生成走大模型，和演示单元接大模型时同一套接线。评测集全是合成数据，只在本地开发机上跑。

```bash
# 密钥只放本地 .env（已 gitignore）或云端环境的密钥设置；不要写进仓库、命令行历史或云端环境变量
set -a; source .env; set +a      # 提供 RODIT_LLM_BASE_URL / RODIT_LLM_MODEL / RODIT_LLM_API_KEY
export NO_PROXY="$NO_PROXY,模型服务的主机名"   # 本机开着代理时：让模型服务直连（见下面"代理"）

# 1. 试跑：放宽超时，先看模型本身答得怎么样、要多久（这份配置的指纹和部署不同，结果不当上线依据）
uv run python -m rodit.eval --llm env --config ~/rodit-evals/explore.yaml --limit 10 --seed 1
# 2. 按部署配置的全量 + 完整结果 + 成本估算（要当上线依据的就是这一份）
uv run python -m rodit.eval --llm env --config config/rodit.yaml --json ~/rodit-evals/llm.json \
    --price-input 0.002 --price-output 0.008
# 3. §13.5 回归；只跑一个问题类型
uv run python -m rodit.eval --llm env --config config/rodit.yaml --baseline ~/rodit-evals/llm.json --tolerance 0.02
uv run python -m rodit.eval --llm env --config config/rodit.yaml --intent size_advice
```

试跑用的配置放仓库外，只写要改的项，其余是默认值：

```yaml
# ~/rodit-evals/explore.yaml
decision: {timeout_s: 8}          # 路由 / 质检单次调用的超时（部署是 1 秒，§12.1）
generation: {timeout_s: 8}        # 生成（部署是 2.5 秒）
latency: {message_budget_s: 30}   # 每条消息的时间预算（部署是 4.5 秒）
llm:
  extra_body: {thinking: {type: disabled}}   # 模型默认开思考模式时要关掉；写法因模型服务而异，见下
```

- **模型服务特有的参数**写在配置的 `llm.extra_body` 里，原样并入每次请求。现在境内的模型大多默认开思考模式，
  不关的话回答被 max_tokens 截断或为空、耗时也长；关法因模型服务而异（常见的是
  `{thinking: {type: disabled}}` 或 `{enable_thinking: false}`，以模型服务的文档为准，`config/rodit.yaml` 里有
  注释掉的例子）。这一项算进配置指纹：评测时关了思考，部署的配置里也要有同样的一行；不放密钥；
  客户端自己填的字段（model、messages、temperature、max_tokens、stream、response_format、logprobs、
  top_logprobs）不能写在这里。接口地址里不能带用户名密码（`https://用户:密码@主机` 的写法），认证只用 API key。
- **连通性检查**：开跑前按这次评测的配置把要发的请求各试一次（路由的 3 票、质检 1 次、生成 1 次，合成数据，
  不计入用量）。哪一次调用失败、回了空文本、被 max_tokens 截断或读不出来，就不开跑（退出码 2），说明里有是哪种
  请求、失败原因（HTTP 状态码和模型服务给的错误代码）、实测耗时和配置的超时、下一步怎么办；通过时打印各环节的
  耗时（超过超时的 70% 会提醒）和请求是否经代理。报告和日志里没有密钥，接口地址只显示主机名。
  `--skip-preflight` 跳过检查（只在调试时用）。
- **超时**：评测里的每个请求带的是各环节自己的超时（`decision.timeout_s`、`generation.timeout_s`）和每条消息的
  时间预算（`latency.message_budget_s`），`RODIT_LLM_TIMEOUT_S` 管不到它们。要放宽只能换 `--config`，而换了
  配置指纹就变了：放宽超时跑出来的数字用来了解模型，不能当上线依据。
- **运行健康度**：调用失败、被截断、读不出时链路按 §12.2 降级，评测照样跑完，但那组数字量的是兜底逻辑，不是
  模型。报告里（文本和 JSON 的 `health`）写明：路由退回关键词规则、路由丢票、质检判断层不可用、生成没有可用
  草稿、引擎异常各有几条，每个用途的调用里失败（按原因分开，如 `http_429`、`http_400:<错误代码>`、
  `timeout`）、被截断、为空、读不出的各有几次，实际做出路由的是哪个判断层。任何一项超过上限（2%，路由丢票
  10%，至少容许 1 条）这一轮就不算数：报告开头和上线门槛旁边写明原因，退出码 3，JSON 照常写出。
  `--allow-degraded` 不因此返回 3（JSON 里的 `valid` 照旧是 false）；`--max-degraded 0.05` 改这一次的上限。
  连续 8 条用例的大模型调用没有一次用得上（调用失败，或有响应但被截断、为空、读不出）时提前停下（退出码 2，不出报告）；`--min-interval 秒数` 让相邻两条用例的
  开始至少隔这么久（限流时用，等待不计入耗时）。
- **报告**在离线评测的各节（§13.2 评分与上线门槛、§13.3 阈值校准）之外，还有 §12.1 耗时（各环节和端到端的
  P50 / P95，对照目标；各环节有多少调用到超时才结束；有用例被时间预算截断或有调用超时时端到端不写"达标"，
  那个数被截短了；耗时是从这台机器、经这条网络路径测的）、§14 用量（按路由 / 生成 / 质检分的 token 和失败
  次数，失败的调用不在合计里、另给估算；接口没返回用量、回显的模型名和配置的不一样都会写明；给了单价时的
  成本，按 §14 示例口径折算的每店月用量）、降级统计；失败清单里有模型草稿和质检原因（已脱敏）。§13.3 的阈值
  校准只用配置的判断层做出的路由（退回关键词规则的用例不参与，报告里写明排除了几条）。
- **代理**：客户端走环境里配置的 HTTP(S) 代理。本机开着代理时，模型服务的请求多绕一跳，§12.1 的耗时和 1 秒的
  路由超时都会受影响：把模型服务的主机名加进 `NO_PROXY`（或 `NO_PROXY='*'`；另设了小写的 `no_proxy` 时以小写
  的为准）。只删掉 `HTTPS_PROXY` 不可靠：macOS 上 urllib 会退回系统的代理设置。
- 单价是每 1000 token 的价格，也可以用环境变量 `RODIT_LLM_PRICE_INPUT_PER_1K` / `RODIT_LLM_PRICE_OUTPUT_PER_1K`；
  不给只报 token。
- `--config` 不给时用环境变量 `RODIT_CONFIG` 指的文件；两样都没有就按内置默认值跑，并提醒：配置指纹和用
  `config/rodit.yaml` 的部署对不上（那份文件多了店铺默认值两节），结果不能当上线依据。
- `--json out.json` 写完整结果（逐条的类型、评分、耗时、用量；放仓库外，不提交），路径开跑前先验证写得进去，
  跑完先写 JSON 再出文本报告；`--json` 不带路径时输出到标准输出。
  `--baseline` 只比较两次都有的用例：类型准确率、路由、事实正确、合规满分率（全部和其中的测试集）任一项比基线低超过
  容差（默认 0，§13.5 指标不下降；真实模型的路由投票有随机性，回归时给容差）时退出码为 1。
- 退出码：0 正常；1 指标比基线下降超过容差；2 没有跑或没有跑完（配置错误、`--json` 的路径写不进去、连通性检查
  没过、连续 8 条用例的调用没有一次用得上而停下）；3 这一轮没有有效评到模型（报告和 JSON 照常写出，同时有回归也是 3）；
  130 被中断（结果没有保存）。`--seed` 只决定 `--limit` 抽哪几条，不发给模型（路由投票要靠采样的随机性，§6.4）。
- **当上线依据**（§13.2、§13.4）：评测用的 `--config` 要和部署的是同一份（镜像里是 `/app/config/rodit.yaml`），
  `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` 也一致；然后在单元里（连得上它的数据库的地方）跑
  `python -m rodit.ops record-eval --report llm.json`。部分评测（`--intent` / `--limit`）、条数和评测集对不上的、
  没有运行健康度或健康度不过关的报告它不收（退出码 2，写明原因）。`llm.public_name` / `llm.filing_no` 也在配置
  指纹里：先填好再跑要当依据的那一轮，否则填的那天之前的结论就对不上了。

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
| `onboard-shop --shop-id … (--token-stdin \| --token-file 文件) [--granted-at 平台开通时间] …` / `refresh-tokens` / `purge-expired [--dry-run]` / `revoke --shop-id …` / `set-status --shop-id … --status …` | §8.4 店铺生命周期（见下文"上线就绪"）。除了 `--dry-run` 都要拿单副本租约：`serve` 在跑时拒绝执行（退出码 2），日志写明改用哪个接口或 `serve` 里的哪个定时任务 |
| `record-eval --report 结果.json` / `acceptance-report --since … [--json]` / `compliance-scan [--week …]` | §13.2、§13.4、§15 上线门槛的数据和 Phase 1 验收报告 |
| `sync-platform-metrics [--days N] [--shop-id …]` | §12.4 从平台指标接口同步客服考核数据（见"Phase 2 / 3"） |

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
| `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` / `RODIT_LLM_TIMEOUT_S` | | 境内已备案大模型。也可以写在配置文件 `llm` 段：`base_url` 和 `model` 都写了才用配置文件，否则两项都取环境变量（启动检查、就绪检查、评测指纹同一条规则） |
| `RODIT_DB_STATEMENT_TIMEOUT_MS` / `RODIT_DB_LOCK_TIMEOUT_MS` | 2000 / 1000 | 请求事务的语句 / 等锁超时（§12.1、§12.2） |
| `RODIT_CONFIG` | 镜像里 `/app/config/rodit.yaml` | 引擎配置 |
| `RODIT_METRICS_PUBLIC` | `false` | `/metrics` 不要令牌（只在网络策略已经限制访问时用） |
| `RODIT_HOST` / `RODIT_PORT` | 镜像里 `0.0.0.0` / `8000` | 监听地址 |
| `RODIT_LOG_LEVEL` / `RODIT_LOG_FORMAT` | `INFO` / `json` | 日志 |
| `RODIT_MIGRATE_ON_START` | `false` | `serve` 前先迁移（服务进程需要属主权限，不推荐） |
| `RODIT_AUTH_EXPIRY_WARN_DAYS` / `RODIT_AUTH_EXPIRY_CHECK_S` | 7 / 300 | 授权即将过期告警的窗口、检查间隔（`serve` 的后台线程；0 = 不检查） |
| `RODIT_READINESS_CACHE_S` | 2 | `/readyz` 结果缓存（后台刷新不用缓存） |
| `RODIT_READINESS_REFRESH_S` | 10 | 后台刷新就绪检查的间隔（每轮都真的检查；0 = 只由 `/readyz` 刷新）；`/metrics` 只输出上次的结果，超过 max(60, 3 × 间隔) 秒没更新按不就绪报（间隔为 0 时是 60 秒 + 缓存期） |
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
  用于商家控制台的商品同步（§5.2.3），不给时同步接口返回 503；
  `vision` 实现 `ImageUnderstanding`（平台云内的视觉模型，买家图片和售后凭证，返回的文字要先打码；不给时没有图片理解）；
  `metrics_source` 实现 `PlatformMetricsSource`（平台客服考核数据，只读；不给时报表只有商家登记的数）。
- **令牌文件**：由平台的登录体系（工作台子账号等）签发令牌，写进挂载的密钥文件（格式见 `rodit.ops.tokens`）。
- **数据库账号**：属主账号给 `migrate` 任务；服务账号 `GRANT rodit_app TO <账号>`；
  离线训练 / 重算任务单独一个账号 `GRANT rodit_trainer TO <账号>`（`RODIT_TRAINER_DSN`）。
- 大模型：平台云内可调用的境内已备案模型（D5），OpenAI 兼容接口。

### 探针与指标（§12.4）

- `/healthz`：存活，不查依赖（和 `/metrics` 一样是异步处理函数，不和业务请求抢工作线程：数据库不通、业务请求把线程占满时也能答）；
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
  不进命令行参数；`--granted-at` 填服务市场订单上的平台开通时间，不填时不记开通时间——不拿本机时间冒充平台时间，
  否则在它之前发生、之后才送到的取消订购会被当成旧事件忽略）。开通时权限按 `tenancy.onboarding_phase` 核对
  （1 = MVP 只读；写售后要到二期）。
- 续期：给 `tenancy.refresh_ahead_hours` 内到期、有续期令牌的店铺续期；续不了的标记"要重新授权"，
  之后的处理一律拒绝。续期令牌同样应用内加密（附加认证数据和访问令牌不同）。每份授权有一个不重复的写入版本：
  续期结果按版本条件写回，"要重新授权"只标记被拒的那一份（平台拒绝旧令牌时，已经换上的新授权不受影响）；
  平台云代码调 `TokenProvider.report_rejected(shop_id, token)` 时带上被拒的令牌。
- 取消订购 / 授权过期：取消推送立即删除该店铺在本单元的数据；过期超过 `tenancy.purge_after_expiry_days`
  （默认 30 天，【待定：委托处理协议】）的按约定删除。删除之前先等这家店留在后台的线程（到截止时间没做完的分条、
  质检、生成、平台查询）做完，最多 30 秒，它们迟到的缓存、用量记录一并删掉。删令牌之前先留一条开始清理的标记
  `shop.purge_started`（系统事件，只有店铺 ID、原因和 `event_at`），删除后留一条不带店铺数据的系统事件
  `shop.purged`（只有店铺 ID、原因、各类数据的条数和触发它的平台事件时间 `event_at`）；商家令牌调
  `POST /v1/shops/{shop_id}/revoke` 也写这条（原因 `merchant_revoke`）。
- 订购事件按平台的事件时间排先后：每家店记着已经处理过的最新事件时间（授权的开通时间、`shop.purged` 的
  `event_at`；开通还要比 `shop.purge_started` 的 `event_at`，清理没做完时被删的那份授权晚到也不写回来），
  不晚于它的开通或取消（重放的、乱序迟到的）一律 `ignored`，判断和写入都在这家店的写方闸门里做。令牌文件里这家店的访问令牌要另外删掉（清理过、没有再开通的
  店铺不再计入授权过期告警）。`revoke --shop-id` 手动取消，`set-status` 切换试用 / 正式 / 停用。
- 续期和过期清理在 `serve` 进程里按间隔跑（`RODIT_TOKEN_REFRESH_INTERVAL_S` 默认 1 小时、
  `RODIT_PURGE_EXPIRED_INTERVAL_S` 默认 6 小时，起来 1 分钟后跑第一轮），不用另外排定时任务：取消订购的闸门
  （等在途处理做完再删）、工具缓存、适配器去重记录都在 `serve` 进程内，另起一个进程删数据时 `serve` 手上的消息
  照样把推荐和审计写进来、再也没有人删。清理中途出错或进程被杀的店铺（令牌已删、数据没删完、没有
  `shop.purged`）每轮先接着删完（按开始清理的标记找，重启之后也一样；原因、`event_at` 沿用，期间重新订购了的
  不删），`purge-expired --dry-run` 也数出这些店铺；出错只记日志和
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
  店铺禁用词，只输出条数和 ID，不输出原文。命中的是疑似违规（候选）：上线状态和验收报告里显示"未复核"，
  复核人员 `POST /v1/shops/{shop_id}/rollout/compliance-reviews` 确认或标记误报（审计 `compliance.reviewed`，
  不记原文），只有确认的才算违规。编辑后发送的推荐和它发出去的那条消息只算一次、记在推荐 ID 上：展示在窗口
  开始之前、发送在窗口里的也一样（配对时往前多看 7 天），所以上线状态、验收报告列出的 ID 都能确认；
- 某周登记的"0 条"要在那一周结束之后登记才算那一周的依据，周中登记的是"暂定"，要复核人员周末后再确认一次；
- `python -m rodit.eval --json 结果.json` 带配置指纹；`record-eval --report 结果.json` 把结论记下来，
  指纹和运行中的配置一致时才算测试集门槛的依据（离线替身的结论不算）。真实大模型跑的报告还要带运行健康度
  （`health`）：没有这一块的（旧版本生成、或没跑完）、条数和评测集对不上的、按固定上限（2%，路由丢票 10%）
  重新算下来大模型没有被有效评到的，`record-eval` 不收（退出码 2，逐条写明原因；评测时用 `--max-degraded`
  放宽过上限也一样）；收下的结论里带着健康度的计数，上线状态里看得到。指纹包括配置（含 `llm.extra_body`）、判断层问题版本、
  路由用的判断层（分类器时还有模型文件的 SHA-256）、大模型名称和接口主机、生成环节的版本（`rodit.generation`
  整个包和依据行格式、类型名等进提示词的 `core` 模块）、判断层提示词的版本（`decision.llm` 的提示词、
  `decision.voting` 的输出解析、路由 / 质检的 state）：换模型（哪怕只改环境变量）、换分类器文件、改提示词都要
  重新评测，模块清单见 `rodit.eval.fingerprint`（清单以外的代码改动只记代码版本、不比较）。上线状态按 `serve`
  启动时实际加载的分类器核对：原地覆盖模型文件（`train-classifier --overwrite`）后不重启，给新文件跑的评测不算。
  `record-eval` / `acceptance-report` 要和 `serve` 用同样的 `RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL`（按分类器
  路由时还有 `RODIT_CLASSIFIER_PATH`，这两个命令读的是文件），否则测试集一项显示"未核对"；
- `acceptance-report --since … [--until …] [--json]`：按店铺和全单元汇总 Phase 1 验收（§13.2 门槛、连续两周
  采纳率 ≥ 60%、P95 ≤ 5 秒、违规为 0、每周复核事实正确和合规满分率 ≥ 95%），全部通过退出码 0，否则 1 并写明哪一项；
- 采纳率不再把全自动发出的回复算进去，单列"全自动"。

**运维加固（§9.1、§12.3、§12.4）**：
- 单副本租约：`serve` 启动时在 PostgreSQL 上拿按平台的会话级咨询锁，已有进程持有时拒绝启动（退出码 2，日志写明
  持有者）；`/readyz` 多一项 `lease`。会话锁、取消订购的闸门都在进程内，Phase 1 一个平台单元只跑一个副本；
  改数据的运维命令也拿这个租约（见上），运维命令在跑时 `serve` 起不来（退出码 2，命令做完后编排系统重启即可）；
  连接串要直连或走会话模式的连接池（事务模式的连接池拿不住会话锁）。
- 抓取、存活检查不排队：业务接口是同步函数，共用一个线程池，数据库连不通时每个请求都要等连接超时、线程池会排满；
  `/metrics`、`/healthz` 是 async 接口，不占这个线程池，照样马上回答（`/metrics` 只读内存，不查库）。
  就绪结果由后台每轮真的检查一次（不用 `/readyz` 的缓存，缓存调长了也不会误报不就绪）；授权过期检查出错时
  后台任务这一轮记 `error`（`RoditBackgroundTaskFailing`），上次成功的时刻不往前走。
- 大模型出口可注入：平台绑定的 `llm_transport`（和 OpenAI 兼容客户端的 transport 同签名）给奇门 / EGW 等报备通道用，
  计量和就绪检查照旧。`RODIT_LLM_BASE_URL` / `RODIT_LLM_MODEL` 去掉首尾空白再用（从密钥文件注入常带换行），
  组装客户端、启动前检查、就绪检查、评测指纹看到的是同一个值。
- 模型公示：配置 `llm.public_name` / `llm.filing_no`（【待定：D5】），`/v1/me` 返回、`/assist` 页脚显示；
  `llm.require_disclosure: true` 时生产部署缺了拒绝启动。
- 指标：`rodit_tenant_isolation_total{stage}`（串店拦截：`http` 是调用方令牌不是这家店的，`server` 是接口处理中
  服务端发现的，其余是引擎在哪一步拦下的）、平台工具按 `unavailable / throttled / timeout` 分开、
  工具缓存命中 / 过期兜底、`rodit_ready` 和各项就绪检查、后台任务每一轮的结果；告警和预计算规则在 `deploy/prometheus/`
  （授权即将过期 / 已过期、任何串店、端到端 P95 > 5 秒、工具和大模型失败率、降级突增、就绪失败、抓不到指标、
  后台任务出错）。Prometheus 抓取任务的 `job_name` 要叫 `rodit`（`RoditScrapeDown` 按它找目标）。

## Phase 2 / 3：售后审核工作台、平台指标接入前后对比、全自动与半自动、图片理解、判断层对比（方案 §15）

**售后审核工作台（§7.3.2、§15 Phase 2）**：网页 `/aftersale`（客服、店长、商家；和 `/assist`、`/console` 同样的
登录和安全约束）。

| 接口 | 作用 |
|---|---|
| `GET /v1/shops/{shop_id}/aftersales` | 待审核队列：平台待审核的售后单 + 本地还没结束的审核记录，按紧急程度排；只读，不跑规则 |
| `POST /v1/shops/{shop_id}/aftersales/review-pending` | 批量"拉取并跑规则"（最多 20 张，逐张报告结果，单张失败不影响别的） |
| `GET /v1/shops/{shop_id}/aftersales/{aftersale_id}` | 详情：平台快照、命中规则和理由、本平台支持的操作（拼多多没有拒绝接口）、能否确认（R6 店长）、有没有写售后权限 |
| `GET /v1/shops/{shop_id}/aftersales/stats` | 各状态条数、按规则的待处理数、平均审核时长（申请 → 经工作台人工确认执行；抖店"售后平均审核时长"的参考） |

建议、协商、确认执行仍是原来的 `aftersales/{id}/review | negotiate | negotiation-done | confirm`：
确认前重读平台状态、核对状态版本、带幂等键，写操作一律要人确认。统计不读平台：申请时间取确认执行时写进
`aftersale.confirm` / `aftersale.action` 审计的 `aftersale_created_at`（之前确认的单计入"时长未知"）；
列表和统计整批读审核记录。批量跑规则时 R5 凭证的图片识别这一批共用 20 秒预算、超时一次就熔断
（其余单的理由注明"凭证图片未能识别"）。

**平台客服指标接入前后对比（§1.3、§2.1、§12.4、§15 Phase 2）**：每个平台只看本平台的考核指标
（`core.platform_metrics`：淘宝 3 分钟人工响应率；抖店飞鸽平均响应时长、会话不满意率、平台求助率、售后平均审核时长；
拼多多 3 分钟人工回复率）。数据来源按优先级：平台接口（平台绑定 `metrics_source`）> 商家登记 > 系统估算。
接入日期：填写的 > 第一次切到智能辅助 > 这次开通的时间（该店最早的 `auth.granted` 审计，优先用平台事件时间；
续订不改变它）> `ShopAuth.granted_at`（续订会改写，只作兜底）> 未知；接入前后各取 28 天，按样本量加权，给出每项的变化和
"较接入前改善 / 有好有坏 / 变差 / 数据不足"。

| 接口 / 命令 | 角色 |
|---|---|
| `GET /v1/shops/{shop_id}/reports/platform-metrics?go_live=&window_days=` | 店长、商家、复核人员 |
| `GET /v1/shops/{shop_id}/platform-metrics?since=&until=`（原始数据） | 同上 |
| `POST /v1/shops/{shop_id}/platform-metrics`（商家登记一天的数） | 商家 |
| `POST /v1/shops/{shop_id}/platform-metrics/sync?days=`（同步平台数据） | 店长、商家 |
| `python -m rodit.ops sync-platform-metrics [--days N] [--shop-id ID …]` | 运维；不拿单副本租约（原因见模块说明），每天一次 |

存储是迁移版本 6 的 `platform_metric` 表（行级安全按店铺隔离，离线训练账号读不到；取消订购一并删除）。
商家控制台「报表」页有"平台客服指标 · 接入前后对比"一节。

**全自动先在非工作时间试行（§15 Phase 3）**：全自动除了总开关（`auto_reply_enabled`，要平台准入）和类型阈值，
还看生效时段 `auto_reply_schedule`：默认 `off_hours`，只在店铺客服工作时间（`service_hours`，北京时间，
如 `09:00-23:00`，可跨午夜）以外自动回复；没填工作时间就不自动（`always` 才全天）。被时段拦下的消息
转给人工、照常出推荐，`engine.result` 记 `auto_gate`。工作时间商家可以在控制台「设置」里改，问卷第 13 题
（"早9点到晚11点"之类）也会写进去；生效时段只能由我们改。工作时间只认"在岗时段"，读不准就不认（宁可不自动）：
带休息、午休、不在、下班、机器人、值班、以外、除了……这类说明的拒绝，"全天 / 24 小时"和具体时段混写的拒绝，
"晚上 0 点 / 12 点"按午夜算；问卷第 13 题读不懂时清空工作时间（不能全自动），接口返回 `warnings` 提醒商家去「设置」填。

**半自动（§5.6、§13.4）**：复核人员按类型开启（`POST /v1/shops/{shop_id}/rollout/semi-auto`，商家和复核人员都能关）：
最近 2 个完整周、每周至少 10 组推荐、原样采纳率 ≥ 90% 才能开，没过门槛要强制并写原因；高风险类型永远不能开。
开了的类型，置信度 ≥ `routing.semi_auto_threshold`（0.9）、规则 6 正常回答、质检一次就通过（被质检拦过、重写后
才通过的不算）、没有提醒和降级的第一条推荐
带 `prefill`：工作台插件直接填进输入框，客服按发送（不是自动发送）。上线状态（`rollout/status`）带各类型的门槛
和全自动的生效时段；控制台「上线」页可以操作。

**图片理解（§15 Phase 3）**：买家发来的图片（模式 A / B，最多 3 张）交给平台云内的视觉模型（平台绑定 `vision`，
`core.ports.ImageUnderstanding`，返回的文字必须已打码），每条消息另有 1.5 秒的识别预算。识别结果写成打码后的
描述进路由和生成（物流 / 订单截图和其他图片可能有收件人姓名，进提示词和推荐记录的只有类型名，识别摘要只在
给客服的提醒里）：破损、质量瑕疵、错发漏发 → 质量问题（推荐里先回应照片、引导在订单页申请售后并上传照片，
不承诺处理结果）；尺码截图里的身高体重 → 尺码推荐；物流截图 → 物流问答；商品图 → 澄清是哪一款。
每条带图片的消息都给客服一条提醒（识别摘要"仅供参考"，或"没能识别，请自己看图"），所以不会被半自动填入、
也不会被全自动发出。识别失败、超时记 `vision` 降级，照常出推荐。售后 R5（质量问题）有凭证图片时，审核建议的
理由后面加一句"图片理解（仅供参考）：…"，最多看 6 张，建议仍由客服决定。

**判断层离线对比（§6.3、§6.4、§13.3；§15 Phase 0 / 3）**：

```bash
uv run python -m rodit.eval.compare                                  # 关键词规则（离线基线）
uv run python -m rodit.eval.compare --classifier model.json --llm env \
    --json ~/rodit-evals/compare.json                                 # 加上分类器、境内大模型
set -a; source .env; set +a                                           # 只在本地开发机：RODIT_JEV_BASE_URL / RODIT_JEV_API_KEY
uv run python -m rodit.eval.compare --providers rules,incloud_llm,jev --llm env --jev env
```

每个判断层收到完全相同的路由请求（§6.5 Jev 格式），报告类型准确率（总体、按类型、dev / test）、macro-F1、
常见混淆、要人工 / 情绪的一致率（数据集没有逐条标注，用路由标注推出的近似标签，报告里注明）、
校准（ECE、Brier、可靠性表）、§13.3 按类型的阈值和覆盖率、耗时和用量。跑了 Jev 时给出 §6.3 条件 4
（"中文测试集上准确率和概率校准达到境内方案的水平"：类型准确率比境内最好的低不超过 1 个百分点、ECE 高出不超过
0.02，可用 `--accuracy-tolerance` / `--ece-tolerance` 调）的结论，
只对照 §6.4 的境内方案（境内大模型 `--llm env`、路由分类器 `--classifier`），关键词规则只是兜底、不作对照：
只加 `--jev env` 时条件 4 判"无法判定"；`--limit` 抽样时也不下结论（数字照列）。macro-F1 按"算对"口径
（答了可接受的备选类型算对）。条件 1–3 需要人工确认。只发合成数据；结论不代表 Jev 可以进生产链路（§6.2）。Jev 的环境变量：
`RODIT_JEV_BASE_URL`（https，不带 `/api/v1/systemone`）、`RODIT_JEV_API_KEY`（只放本地 `.env`）、`RODIT_JEV_TIMEOUT_S`
（默认 10）、`RODIT_JEV_MODEL`（默认 `jev-1.13.0`）。退出码 0 完成、2 配置不对。

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
| `rodit.eval` | 合成评测集（324 条）、§13.2 评分、§13.3 阈值校准；离线替身或真实大模型、耗时与成本、回归比较；每月阈值重算、路由分类器训练；判断层离线对比（含 Jev） | §13、§12.1、§14、§6.3、§6.4 |
| `rodit.showcase` | 客户演示站（独立应用，合成数据 + 离线替身）：产品概览、智能辅助工作台、售后审核、数据看板、合规与部署 | §0、§5、§7.3、§9.1、§11、§12.4 |

## 数据合规（摘自 CLAUDE.md）

- 买家和订单数据只在对应平台云内处理，不出境；本仓库只有**合成数据**（`rodit.testing.fixtures`）。
- 仓库和云端环境变量里没有任何生产密钥；密钥只从部署环境读（见 `rodit.app.production`）。
- 平台开放接口一律用模拟实现；真实实现要在平台云内开发和部署。
- 提示词、判断层请求、审计日志里不放买家昵称、手机号、收件地址（`rodit.core.pii.mask_pii`）。
- 售后写操作一律要客服确认（`HumanConfirmation`），执行前重读平台状态、带幂等键。

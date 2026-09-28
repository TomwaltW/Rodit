-- Rodit 数据库结构，迁移版本 1（方案 §10 数据模型、§9.2 租户隔离、§8.4 授权令牌加密）。
--
-- 由 rodit.storage.db.Database.migrate() 在一个事务里执行，并记入 schema_migrations；
-- 脚本本身也可以重复执行（IF NOT EXISTS、DROP POLICY IF EXISTS），不会报错也不会丢数据。
-- 每个平台单元（聚石塔 / 抖店云 / 多多云）各有一份库（§9.1）。
--
-- 与方案 §10 的差异（按 core.models 调整，见 docs/架构与分工.md 的契约约定）：
-- - 业务表都有 shop_id NOT NULL（audit_log 例外：没有店铺归属的系统事件 shop_id 为空）；
--   不设指向 shop 的外键：各仓储按协议独立写入（与内存实现语义一致），
--   店铺数据的删除由各仓储的 purge_shop 负责（§8.4）。
-- - conversation 的主键是 (shop_id, id)：不同店铺可以有同名会话 ID；
--   message 用 (shop_id, conversation_id) 指向会话（延迟检查，先写消息再补会话）。
-- - shop.config 存 Shop.config_overrides（只存店铺覆盖项，生效配置由 core.config 计算，§9.3）。
-- - shop_auth.access_token 是 pgcrypto 对称加密后的 BYTEA（方案写 TEXT「加密存储」）。
-- - knowledge_item 增加 item_id / intent / created_at；向量用 REAL[] 代替 VECTOR(1024)：
--   沙箱和部分平台 RDS 没有 pgvector，单店知识量小，候选在应用内算余弦（见 repositories.py）。
--   平台 RDS 支持 pgvector 后换成 VECTOR(维度) + HNSW 索引（vector_cosine_ops）。
-- - message 增加 sku_id；suggestion 增加 msg_id / question_text / item_id / kind；
--   aftersale_review 增加 reason / requires_manager / risk_flags / verify_started_at / updated_at，
--   version 从 1 开始（新建即 1，每次写入 +1，乐观锁）。
-- - 新增 knowledge_gap（知识缺失提醒）、labeled_sample（判断层标注集），见 §5.8。
--
-- 行级安全（§9.2）：每张业务表 ENABLE + FORCE ROW LEVEL SECURITY，策略为
--   shop_id = current_setting('rodit.shop_id', true)
-- 应用以 rodit_app 角色访问（Database.session 里 SET LOCAL ROLE），每个事务设置 rodit.shop_id；
-- 没设置店铺时 current_setting 为 NULL 或 ''，与任何行都不相等，读不到数据。
-- 即使仓储代码漏写 WHERE shop_id，也读不到、写不进别的店铺的数据。
--
-- 训练导出（FeedbackRepository.export_training_labels，§6.4）要跨店读 labeled_sample：
-- 用单独的角色 rodit_trainer，只有 labeled_sample 的 SELECT 权限和一条只读策略。
-- 没有用 SECURITY DEFINER 函数：那样 rodit_app 的任何代码路径（包括商家接口）都能调用它；
-- 单独角色的权限在库里一目了然，生产上训练任务用只属于 rodit_trainer 的登录账号，
-- 商家侧服务的登录账号只属于 rodit_app。

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------- 角色（集群级，已存在则跳过）

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rodit_app') THEN
    BEGIN
      CREATE ROLE rodit_app NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    EXCEPTION WHEN duplicate_object OR unique_violation THEN
      NULL;  -- 另一个会话同时创建了
    END;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rodit_trainer') THEN
    BEGIN
      CREATE ROLE rodit_trainer NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    EXCEPTION WHEN duplicate_object OR unique_violation THEN
      NULL;
    END;
  END IF;
  -- 角色可能由运维预先创建：绕过行级安全的角色不能用（§9.2）
  IF EXISTS (
    SELECT 1 FROM pg_roles
    WHERE rolname IN ('rodit_app', 'rodit_trainer') AND (rolsuper OR rolbypassrls)
  ) THEN
    RAISE EXCEPTION 'rodit_app / rodit_trainer 不能是超级用户，也不能带 BYPASSRLS（方案 §9.2）';
  END IF;
END
$$;

-- ---------------------------------------------------------------- 店铺与授权

CREATE TABLE IF NOT EXISTS shop (
  id          TEXT PRIMARY KEY CHECK (id <> ''),     -- 平台内店铺 ID
  platform    TEXT NOT NULL,                          -- taobao / douyin / pdd
  category    TEXT,                                   -- 主营类目
  status      TEXT NOT NULL,                          -- trial / active / suspended
  config      JSONB NOT NULL DEFAULT '{}'             -- Shop.config_overrides（只存覆盖项）
);

CREATE TABLE IF NOT EXISTS shop_auth (
  shop_id       TEXT PRIMARY KEY CHECK (shop_id <> ''),
  access_token  BYTEA NOT NULL,                       -- pgp_sym_encrypt 加密（§8.4）
  scopes        TEXT[] NOT NULL,
  expires_at    TIMESTAMPTZ NOT NULL
);

-- ---------------------------------------------------------------- 知识（§5.2）

CREATE TABLE IF NOT EXISTS shop_policy (
  shop_id     TEXT NOT NULL CHECK (shop_id <> ''),
  key         TEXT NOT NULL,                          -- core.enums.PolicyKey 的取值
  value       TEXT NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (shop_id, key)
);

CREATE TABLE IF NOT EXISTS item_card (
  shop_id     TEXT NOT NULL CHECK (shop_id <> ''),
  item_id     TEXT NOT NULL,
  card        JSONB NOT NULL,                         -- §5.2.2（shop_id / item_id / updated_at 在列上）
  updated_at  TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (shop_id, item_id)
);

CREATE TABLE IF NOT EXISTS knowledge_item (
  id          BIGSERIAL PRIMARY KEY,
  shop_id     TEXT NOT NULL CHECK (shop_id <> ''),
  question    TEXT NOT NULL,
  answer      TEXT NOT NULL,
  source      TEXT NOT NULL,                          -- merchant / adopted / category_template
  status      TEXT NOT NULL,                          -- candidate / approved / retired
  item_id     TEXT,                                   -- 只适用于某个商品的 FAQ
  intent      TEXT,
  approved_by TEXT,
  created_at  TIMESTAMPTZ NOT NULL,
  -- 维度按所选向量模型；平台 RDS 支持 pgvector 时换成 VECTOR(维度)
  embedding   REAL[]
);
CREATE INDEX IF NOT EXISTS knowledge_item_shop_status_idx ON knowledge_item (shop_id, status, id);

-- ---------------------------------------------------------------- 会话（§5.3、§8.3）

CREATE TABLE IF NOT EXISTS conversation (
  shop_id     TEXT NOT NULL CHECK (shop_id <> ''),
  id          TEXT NOT NULL,
  buyer_ref   TEXT NOT NULL,                          -- 买家 ID 哈希
  started_at  TIMESTAMPTZ NOT NULL,
  closed_at   TIMESTAMPTZ,
  PRIMARY KEY (shop_id, id)
);

CREATE TABLE IF NOT EXISTS message (
  id              BIGSERIAL PRIMARY KEY,              -- 写入顺序（同一时间按它排序）
  shop_id         TEXT NOT NULL CHECK (shop_id <> ''),
  msg_id          TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  role            TEXT NOT NULL,                      -- buyer / agent / bot
  text            TEXT,
  item_id         TEXT,
  sku_id          TEXT,
  order_ref       TEXT,
  created_at      TIMESTAMPTZ NOT NULL,
  UNIQUE (shop_id, msg_id),
  FOREIGN KEY (shop_id, conversation_id) REFERENCES conversation (shop_id, id)
    ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX IF NOT EXISTS message_conversation_idx
  ON message (shop_id, conversation_id, created_at, id);

-- ---------------------------------------------------------------- 判断与推荐（§6、§5.6）

CREATE TABLE IF NOT EXISTS decision_log (
  id              BIGSERIAL PRIMARY KEY,
  shop_id         TEXT NOT NULL CHECK (shop_id <> ''),
  conversation_id TEXT,
  stage           TEXT NOT NULL,                      -- route / qa
  provider        TEXT NOT NULL,                      -- incloud_llm / classifier / rules / jev
  questions_ver   TEXT NOT NULL,                      -- 问题定义的版本（core.questions）
  answers         JSONB NOT NULL,
  route_result    TEXT,
  latency_ms      INT,
  created_at      TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_log_shop_time_idx ON decision_log (shop_id, created_at, id);

CREATE TABLE IF NOT EXISTS suggestion (
  id              BIGSERIAL PRIMARY KEY,
  shop_id         TEXT NOT NULL CHECK (shop_id <> ''),
  conversation_id TEXT NOT NULL,
  msg_id          TEXT,                               -- 触发推荐的买家消息
  question_text   TEXT,                               -- 已做个人信息最小化
  item_id         TEXT,
  intent          TEXT NOT NULL,
  text            TEXT NOT NULL,
  kind            TEXT NOT NULL,                      -- faq / template / llm
  evidence_ids    TEXT[] NOT NULL DEFAULT '{}',
  qa_scores       JSONB,
  status          TEXT NOT NULL,                      -- shown / sent / edited_sent / rejected / expired / shadow
  edited_text     TEXT,
  reject_reason   TEXT,
  agent_id        TEXT,
  latency_ms      INT,
  created_at      TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS suggestion_shop_time_idx ON suggestion (shop_id, created_at, id);
CREATE INDEX IF NOT EXISTS suggestion_conversation_idx
  ON suggestion (shop_id, conversation_id, created_at, id);

-- ---------------------------------------------------------------- 售后审核（§7.3.2）

CREATE TABLE IF NOT EXISTS aftersale_review (
  id                BIGSERIAL PRIMARY KEY,
  shop_id           TEXT NOT NULL CHECK (shop_id <> ''),
  aftersale_id      TEXT NOT NULL,                    -- 平台售后单 ID
  platform_status   TEXT NOT NULL,
  state             TEXT NOT NULL,                    -- 见 core.enums.ReviewState
  matched_rule      TEXT,                             -- ",".join(matched_rules)，如 "R3,R6"
  suggestion        TEXT,                             -- core.enums.ReviewAdvice
  reason            TEXT,
  requires_manager  BOOLEAN NOT NULL DEFAULT false,   -- R6
  risk_flags        TEXT[] NOT NULL DEFAULT '{}',     -- R7 等
  verify_started_at TIMESTAMPTZ,                      -- R4
  confirmed_by      TEXT,
  action_result     JSONB,
  version           INT NOT NULL DEFAULT 1 CHECK (version >= 1),  -- 乐观锁
  updated_at        TIMESTAMPTZ,
  UNIQUE (shop_id, aftersale_id)
);
CREATE INDEX IF NOT EXISTS aftersale_review_state_idx ON aftersale_review (shop_id, state);

-- ---------------------------------------------------------------- 审计（§4.1 原则 6、§12.4）

CREATE TABLE IF NOT EXISTS audit_log (
  id          BIGSERIAL PRIMARY KEY,
  shop_id     TEXT CHECK (shop_id <> ''),             -- 为空：没有店铺归属的系统事件
  actor       TEXT NOT NULL,                          -- system / agent_id / merchant
  action      TEXT NOT NULL,                          -- core.audit.AuditAction
  detail      JSONB NOT NULL DEFAULT '{}',            -- 不放买家原文
  created_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_log_shop_time_idx ON audit_log (shop_id, created_at, id);
CREATE INDEX IF NOT EXISTS audit_log_shop_action_idx ON audit_log (shop_id, action, created_at);

-- ---------------------------------------------------------------- 反馈闭环的派生数据（§5.8）

CREATE TABLE IF NOT EXISTS knowledge_gap (
  id            BIGSERIAL PRIMARY KEY,
  shop_id       TEXT NOT NULL CHECK (shop_id <> ''),
  intent        TEXT,
  missing       TEXT NOT NULL,                        -- 如 policy:invoice
  question_text TEXT,                                 -- 已做个人信息最小化
  source        TEXT NOT NULL,                        -- engine / feedback
  created_at    TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS knowledge_gap_shop_time_idx ON knowledge_gap (shop_id, created_at, id);

CREATE TABLE IF NOT EXISTS labeled_sample (
  id                BIGSERIAL PRIMARY KEY,
  shop_id           TEXT NOT NULL CHECK (shop_id <> ''),
  text              TEXT NOT NULL,                    -- 已做个人信息最小化
  predicted_intent  TEXT,
  true_intent       TEXT,
  source            TEXT NOT NULL,                    -- feedback_reject / feedback_adopt / manual
  created_at        TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS labeled_sample_shop_time_idx
  ON labeled_sample (shop_id, created_at, id);

-- ---------------------------------------------------------------- 行级安全与授权（§9.2）

ALTER TABLE shop ENABLE ROW LEVEL SECURITY;
ALTER TABLE shop FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS shop_isolation ON shop;
CREATE POLICY shop_isolation ON shop
  USING (id = current_setting('rodit.shop_id', true))
  WITH CHECK (id = current_setting('rodit.shop_id', true));
GRANT SELECT, INSERT, UPDATE, DELETE ON shop TO rodit_app;

DO $$
DECLARE
  t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'shop_auth', 'shop_policy', 'item_card', 'knowledge_item', 'conversation', 'message',
    'decision_log', 'suggestion', 'aftersale_review', 'audit_log', 'knowledge_gap',
    'labeled_sample'
  ] LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format('DROP POLICY IF EXISTS shop_isolation ON %I', t);
    EXECUTE format(
      'CREATE POLICY shop_isolation ON %I'
      ' USING (shop_id = current_setting(''rodit.shop_id'', true))'
      ' WITH CHECK (shop_id = current_setting(''rodit.shop_id'', true))',
      t
    );
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO rodit_app', t);
  END LOOP;
END
$$;

-- 没有店铺归属的系统事件：只能在未设置店铺时写入，应用读不到（list_events 必须带店铺）
DROP POLICY IF EXISTS system_event_insert ON audit_log;
CREATE POLICY system_event_insert ON audit_log FOR INSERT
  WITH CHECK (shop_id IS NULL AND coalesce(current_setting('rodit.shop_id', true), '') = '');

-- 训练导出：rodit_trainer 只能读 labeled_sample（跨店），不能写，也读不到其他表
DROP POLICY IF EXISTS trainer_export ON labeled_sample;
CREATE POLICY trainer_export ON labeled_sample FOR SELECT TO rodit_trainer USING (true);
GRANT SELECT ON labeled_sample TO rodit_trainer;

GRANT USAGE ON SCHEMA public TO rodit_app, rodit_trainer;
GRANT USAGE ON SEQUENCE
  knowledge_item_id_seq, message_id_seq, decision_log_id_seq, suggestion_id_seq,
  aftersale_review_id_seq, audit_log_id_seq, knowledge_gap_id_seq, labeled_sample_id_seq
  TO rodit_app;

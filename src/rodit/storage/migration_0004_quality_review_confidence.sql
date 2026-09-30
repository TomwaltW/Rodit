-- Rodit 数据库结构，迁移版本 4：
-- 1. 每周人工复核（§13.5 每周从推荐中抽 50 条人工复核，按 §13.2 四个维度打分）：quality_review
--    只存推荐 ID 和打分，不复制推荐文字和买家问题（复核时从 suggestion 读）；同一店铺、同一周期
--    （ISO 周，如 2026-W40）、同一推荐只有一条。
-- 2. 每月按采纳反馈重算阈值（§13.5、§13.3 的 (预测类型, 置信度, 真实类型) 三元组）：
--    suggestion.intent_confidence 记路由给这条推荐的置信度，labeled_sample.predicted_confidence
--    由反馈闭环从推荐记录抄过来（训练导出角色 rodit_trainer 只读 labeled_sample，够用）；
--    route_rule 记这条推荐走的 §6.6 路由规则，重算辅助阈值只用规则 6 的样本。
--    旧数据为 NULL（重算时跳过置信度为空的）。
--
-- 版本 1–3 已发布，不再修改。与之前的版本一样可以重复执行（IF NOT EXISTS、DROP POLICY IF EXISTS）。

-- ---------------------------------------------------------------- 置信度

ALTER TABLE suggestion ADD COLUMN IF NOT EXISTS intent_confidence REAL
  CHECK (intent_confidence IS NULL OR (intent_confidence >= 0 AND intent_confidence <= 1));
ALTER TABLE labeled_sample ADD COLUMN IF NOT EXISTS predicted_confidence REAL
  CHECK (predicted_confidence IS NULL OR (predicted_confidence >= 0 AND predicted_confidence <= 1));
ALTER TABLE suggestion ADD COLUMN IF NOT EXISTS route_rule SMALLINT
  CHECK (route_rule IS NULL OR route_rule BETWEEN 1 AND 8);
ALTER TABLE labeled_sample ADD COLUMN IF NOT EXISTS route_rule SMALLINT
  CHECK (route_rule IS NULL OR route_rule BETWEEN 1 AND 8);
CREATE INDEX IF NOT EXISTS labeled_sample_time_idx ON labeled_sample (created_at, id);

-- ---------------------------------------------------------------- 每周人工复核

CREATE TABLE IF NOT EXISTS quality_review (
  id            BIGSERIAL PRIMARY KEY,
  shop_id       TEXT NOT NULL CHECK (shop_id <> ''),
  period        TEXT NOT NULL CHECK (period ~ '^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$'),
  suggestion_id BIGINT NOT NULL CHECK (suggestion_id > 0),
  intent        TEXT NOT NULL,
  status        TEXT NOT NULL CHECK (status IN ('pending', 'scored', 'skipped')),
  scores        JSONB,                                  -- §13.2：factual / relevance / compliance / expression
  reviewer      TEXT,
  note          TEXT,                                   -- 复核人备注（已做个人信息最小化）
  sampled_at    TIMESTAMPTZ NOT NULL,
  reviewed_at   TIMESTAMPTZ,
  UNIQUE (shop_id, period, suggestion_id)
);
CREATE INDEX IF NOT EXISTS quality_review_shop_period_idx ON quality_review (shop_id, period, id);

ALTER TABLE quality_review ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_review FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS shop_isolation ON quality_review;
CREATE POLICY shop_isolation ON quality_review
  USING (shop_id = current_setting('rodit.shop_id', true))
  WITH CHECK (shop_id = current_setting('rodit.shop_id', true));
GRANT SELECT, INSERT, UPDATE, DELETE ON quality_review TO rodit_app;
GRANT USAGE ON SEQUENCE quality_review_id_seq TO rodit_app;

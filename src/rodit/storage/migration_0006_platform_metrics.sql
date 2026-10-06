-- Rodit 数据库结构，迁移版本 6（方案 §12.4 平台客服指标接入前后对比、§15 Phase 2 商家报表）：
-- platform_metric：每家店铺每天每项平台客服考核指标一条，按来源分开存（平台接口同步 platform、
-- 商家手工登记 merchant、系统估算 computed）；同一店铺、指标、日期、来源只有一条，重复写入覆盖
-- value / sample_size / recorded_by / recorded_at（core.ports.PlatformMetricsRepository.upsert）。
-- - value 的单位见 core.platform_metrics（比例 0–1、时长秒或小时），只能是 0 及以上的有限数
--   （PostgreSQL 里 NaN 比任何数都大，value < 'Infinity' 同时挡住 NaN 和无穷大）；
-- - sample_size 是这个数背后的会话 / 售后单数，平台给了才有；
-- - recorded_by 是商家登记人的账号（平台同步、系统估算为空），写入前去掉 NUL 等控制字符；
-- - 不含任何买家数据。取消订购时由 PgPlatformMetricsRepository.purge_shop 分批删除（§8.4）。
--
-- 行级安全（§9.2）和其他店铺表一样：ENABLE + FORCE，shop_isolation 策略按 rodit.shop_id 限定，
-- 只授权给 rodit_app；训练导出角色 rodit_trainer 读不到。
--
-- 版本 1–5 已发布，不再修改。与之前的版本一样可以重复执行（IF NOT EXISTS、DROP POLICY IF EXISTS）。

CREATE TABLE IF NOT EXISTS platform_metric (
  shop_id     TEXT NOT NULL CHECK (shop_id <> ''),
  metric      TEXT NOT NULL CHECK (metric <> ''),     -- core.enums.PlatformMetric
  day         DATE NOT NULL,                          -- 东八区的自然日
  value       DOUBLE PRECISION NOT NULL CHECK (value >= 0 AND value < 'Infinity'::float8),
  source      TEXT NOT NULL CHECK (source IN ('platform', 'merchant', 'computed')),
  sample_size BIGINT CHECK (sample_size IS NULL OR sample_size >= 0),
  recorded_by TEXT,
  recorded_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (shop_id, metric, day, source)
);
CREATE INDEX IF NOT EXISTS platform_metric_shop_day_idx ON platform_metric (shop_id, day);

COMMENT ON TABLE platform_metric IS
  '平台客服考核指标（§12.4 接入前后对比）：店铺、指标、日期、来源各一条，不含买家数据';

ALTER TABLE platform_metric ENABLE ROW LEVEL SECURITY;
ALTER TABLE platform_metric FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS shop_isolation ON platform_metric;
CREATE POLICY shop_isolation ON platform_metric
  USING (shop_id = current_setting('rodit.shop_id', true))
  WITH CHECK (shop_id = current_setting('rodit.shop_id', true));
GRANT SELECT, INSERT, UPDATE, DELETE ON platform_metric TO rodit_app;

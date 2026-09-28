-- Rodit 数据库结构，迁移版本 2：售后审核记下建议所依据的平台状态版本（§7.3.2 调用前重新读取）。
--
-- 版本 1（schema.sql）已发布，不再修改：已经执行过版本 1 的库不会重跑它，
-- CREATE TABLE IF NOT EXISTS 也不会给已有的表补列，所以新列必须在新版本里 ALTER。
-- 与版本 1 一样可以重复执行（IF NOT EXISTS）。

ALTER TABLE aftersale_review ADD COLUMN IF NOT EXISTS platform_status_version TEXT;

COMMENT ON COLUMN aftersale_review.platform_status_version IS
  '建议所依据的平台售后单状态版本；确认时平台版本已变则要求刷新';

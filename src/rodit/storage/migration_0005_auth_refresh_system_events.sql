-- Rodit 数据库结构，迁移版本 5（方案 §8.4 店铺授权续期 / 取消订购、§12.4 授权告警、§13.5 回归结论）：
-- 1. shop_auth 加续期令牌和续期状态：refresh_token 和 access_token 一样在应用里用 AES-256-GCM
--    加密（附加认证数据是店铺 ID、一个 NUL 字节、"refresh"：TEXT 里不会有 NUL，所以和任何店铺
--    access_token 的附加认证数据都不相同；挪到别的店铺、或当成 access_token 都解不开），
--    refresh_expires_at 是续期令牌的有效期，refreshed_at 是最近一次续期成功的时间，
--    needs_reauth 表示续期被拒 / 平台拒绝了当前令牌，要商家重新授权（授权检查不通过）。
-- 2. rodit_auth_expiring(before)：续期任务、过期清理要跨店列出"快到期的店铺"，而行级安全不许
--    rodit_app 跨店读 shop_auth。这个 SECURITY DEFINER 函数只返回店铺 ID 和到期时间（不返回令牌，
--    也不返回别的列），固定 search_path，属于第一次执行迁移的账号，只有 rodit_app 能执行，
--    而且只能在不设店铺范围的事务里调用（rodit.shop_id 为空；设了店铺范围时报 insufficient_privilege：
--    商家请求等店铺范围内的代码不能借它列出别的店铺）。
--    函数以属主身份读表：生产上属主不是超级用户，也受 FORCE 行级安全约束，所以给属主加一条
--    只读策略，而且只在这个函数执行期间生效（函数开头打开 rodit.auth_expiring_scan，返回前恢复；
--    出错时随事务回滚）：属主在函数外照旧受行级安全约束；rodit_app 自己设这个开关也没用（策略
--    只给属主）。属主本来就能改表的行级安全设置，这条策略没有扩大它的权限。
--    策略按函数的属主建，不按执行脚本的账号：别的账号（如 DBA 用超级用户）重新执行本脚本时，
--    CREATE OR REPLACE 不改函数的属主，策略也要留在属主身上，否则函数静默地什么都读不到。
-- 3. audit_log 的 system_event_select 策略：不设店铺范围时，应用能读 shop_id 为空的系统事件
--    （取消订购的清理记录 shop.purged、评测结论 eval.verdict）；设了店铺范围时读不到任何系统事件，
--    店铺的事件照旧只有本店能读。
--
-- 版本 1–4 已发布，不再修改。与之前的版本一样可以重复执行（IF NOT EXISTS、CREATE OR REPLACE、
-- DROP POLICY IF EXISTS）。

-- ---------------------------------------------------------------- 续期令牌

ALTER TABLE shop_auth ADD COLUMN IF NOT EXISTS refresh_token BYTEA;
ALTER TABLE shop_auth ADD COLUMN IF NOT EXISTS refresh_expires_at TIMESTAMPTZ;
ALTER TABLE shop_auth ADD COLUMN IF NOT EXISTS refreshed_at TIMESTAMPTZ;
ALTER TABLE shop_auth ADD COLUMN IF NOT EXISTS needs_reauth BOOLEAN NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS shop_auth_expires_idx ON shop_auth (expires_at, shop_id);

COMMENT ON COLUMN shop_auth.refresh_token IS
  '续期令牌，应用内 AES-256-GCM 加密（附加认证数据：店铺 ID、NUL 字节、refresh）';
COMMENT ON COLUMN shop_auth.needs_reauth IS
  '续期被拒或平台拒绝了当前令牌：停止处理，等商家重新授权';

-- ---------------------------------------------------------------- 快到期的店铺（跨店，只有 ID 和时间）

CREATE OR REPLACE FUNCTION rodit_auth_expiring(before TIMESTAMPTZ)
RETURNS TABLE (shop_id TEXT, expires_at TIMESTAMPTZ)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
  previous TEXT := current_setting('rodit.auth_expiring_scan', true);
BEGIN
  IF coalesce(current_setting('rodit.shop_id', true), '') <> '' THEN
    RAISE EXCEPTION 'rodit_auth_expiring 只能在不设店铺范围的事务里调用'
      USING ERRCODE = 'insufficient_privilege';
  END IF;
  PERFORM set_config('rodit.auth_expiring_scan', 'on', true);
  RETURN QUERY
    SELECT a.shop_id, a.expires_at
      FROM public.shop_auth a
     WHERE a.expires_at <= $1
     ORDER BY a.expires_at, a.shop_id COLLATE "C";
  PERFORM set_config('rodit.auth_expiring_scan', coalesce(previous, ''), true);
END
$$;

COMMENT ON FUNCTION rodit_auth_expiring(TIMESTAMPTZ) IS
  '到期时间不晚于 before 的店铺（只有店铺 ID 和到期时间，不含令牌）；续期、过期清理用';

REVOKE ALL ON FUNCTION rodit_auth_expiring(TIMESTAMPTZ) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION rodit_auth_expiring(TIMESTAMPTZ) TO rodit_app;

-- 函数属主读 shop_auth 的只读策略：只在 rodit_auth_expiring 执行期间生效。授给函数的属主
-- （第一次执行时就是执行迁移的账号；别的账号重新执行本脚本时属主不变，策略也不变）。
-- 超级用户本来就绕过行级安全，这条策略对它没有影响
DO $$
BEGIN
  EXECUTE 'DROP POLICY IF EXISTS auth_expiring_definer ON shop_auth';
  EXECUTE format(
    'CREATE POLICY auth_expiring_definer ON shop_auth FOR SELECT TO %I'
    ' USING (current_setting(''rodit.auth_expiring_scan'', true) = ''on'')',
    (SELECT pg_get_userbyid(p.proowner) FROM pg_proc p
      WHERE p.oid = 'rodit_auth_expiring(timestamptz)'::regprocedure)
  );
END
$$;

-- ---------------------------------------------------------------- 系统事件（shop_id 为空）

DROP POLICY IF EXISTS system_event_select ON audit_log;
CREATE POLICY system_event_select ON audit_log FOR SELECT
  USING (shop_id IS NULL AND coalesce(current_setting('rodit.shop_id', true), '') = '');

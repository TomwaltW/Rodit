#!/bin/sh
# 本地试用（deploy/docker-compose.yml）：库第一次初始化时建应用的登录账号。
# rodit_login：能登录、非超级用户、不带 BYPASSRLS，属于 rodit_app（Database.session 里
# SET LOCAL ROLE rodit_app，行级安全生效，方案 §9.2）。表和策略由 rodit 的迁移建（init 服务）。
# 密码来自 deploy/.env（RODIT_DB_APP_PASSWORD），用 psql 变量传入，不出现在命令行参数里。
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v app_password="$RODIT_DB_APP_PASSWORD" <<'SQL'
CREATE ROLE rodit_app NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
CREATE ROLE rodit_login LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE
  PASSWORD :'app_password';
GRANT rodit_app TO rodit_login;
SQL

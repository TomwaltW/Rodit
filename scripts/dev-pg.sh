#!/usr/bin/env bash
# 本地（含云端沙箱）开发用 PostgreSQL：启动服务，并给当前系统用户建一个同名超级用户角色，
# 方便通过 unix socket 免密连接。每个云端会话都要重新执行，数据不跨会话保留。
# 只用于开发和测试；PostgreSQL 测试（@pytest.mark.pg）连不上时会自动跳过。
set -euo pipefail
service postgresql start >/dev/null
for _ in $(seq 1 20); do
  su postgres -c "pg_isready -q" && break
  sleep 0.5
done
ME="$(id -un)"
su postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${ME}'\"" | grep -q 1 \
  || su postgres -c "createuser -s ${ME}"
su postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${ME}'\"" | grep -q 1 \
  || su postgres -c "createdb -O ${ME} ${ME}"
echo "PostgreSQL 就绪：psql 直接连接即可（用户 ${ME}，unix socket）"

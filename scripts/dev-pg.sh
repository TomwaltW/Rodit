#!/usr/bin/env bash
# 本地（含云端沙箱）开发用 PostgreSQL：启动服务，并给当前系统用户建一个同名超级用户角色，
# 方便通过 unix socket 免密连接。每个云端会话都要重新执行，数据不跨会话保留。
# 只用于开发和测试；PostgreSQL 测试（@pytest.mark.pg）连不上时会自动跳过。
#
# macOS 上没有 service / su postgres，改用 Docker 起一个一次性的 PostgreSQL 16 容器 rodit-dev-pg：
# 只发布在本机 127.0.0.1:54329，数据放容器的内存盘（不建卷，容器一停就没了），免密（trust：
# 只有本机连得上，库里只有合成数据）。可以重复执行。测试默认连 unix socket、找不到这个容器，
# 要先执行脚本最后打印的那行 export RODIT_TEST_PG_DSN=…。
# 一个容器同一时间只给一个 pytest 进程用（Docker Desktop 的端口转发每分钟只扛得住约一万条新连接，
# 租约测试还会按连接名杀会话）：几个 worktree 同时跑测试时各起各的，
# RODIT_DEV_PG_NAME=… RODIT_DEV_PG_PORT=… bash scripts/dev-pg.sh。
# 拉不到 Docker Hub 时用 RODIT_DEV_PG_IMAGE 换镜像源。停掉并删除：docker rm -f rodit-dev-pg
set -euo pipefail
if [ "$(uname -s)" = "Darwin" ]; then
  NAME="${RODIT_DEV_PG_NAME:-rodit-dev-pg}"
  docker info >/dev/null 2>&1 \
    || { echo "连不上 Docker（没装或没启动）：先启动 Docker Desktop 再执行本脚本" >&2; exit 1; }
  if docker container inspect "$NAME" >/dev/null 2>&1; then
    docker start "$NAME" >/dev/null # 已在运行时什么也不做
  else
    docker run -d --name "$NAME" -p "127.0.0.1:${RODIT_DEV_PG_PORT:-54329}:5432" \
      --tmpfs /var/lib/postgresql/data -e POSTGRES_HOST_AUTH_METHOD=trust \
      "${RODIT_DEV_PG_IMAGE:-postgres:16}" >/dev/null \
      || { docker rm -f "$NAME" >/dev/null 2>&1; exit 1; } # 没起来（如端口被占）：不留半成品
  fi
  # 走 TCP 探测：镜像初始化时先起一个只听 unix socket 的临时实例，那时还不算就绪
  for _ in $(seq 1 60); do
    docker exec "$NAME" pg_isready -q -h 127.0.0.1 -U postgres && break
    sleep 0.5
  done
  docker exec "$NAME" pg_isready -q -h 127.0.0.1 -U postgres \
    || { echo "容器 ${NAME} 里的 PostgreSQL 没有就绪，看 docker logs ${NAME}" >&2; exit 1; }
  PORT="$(docker port "$NAME" 5432/tcp)"
  PORT="${PORT##*:}"
  echo "PostgreSQL 就绪：容器 ${NAME}，127.0.0.1:${PORT}（用户 postgres，免密）。跑测试前先执行："
  echo "export RODIT_TEST_PG_DSN='host=127.0.0.1 port=${PORT} user=postgres dbname=postgres'"
  exit 0
fi
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

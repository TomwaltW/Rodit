# syntax=docker/dockerfile:1
# Rodit 一个平台单元的镜像（方案 §9.1 单元化部署：一套代码，每个平台云各部署一份）。
#
# 镜像里没有任何密钥：数据库连接串、店铺令牌加密密钥、buyer_ref 的盐、大模型密钥、访问令牌文件
# 都在运行时由平台云的密钥管理注入（环境变量 / 挂载文件），见 README「部署：一个平台单元」。
# 平台只读工具和接入适配器由平台云代码提供（RODIT_PLATFORM_BINDINGS），或只部署模式 C。
#
#   docker build -t rodit:local .
#
# 平台云内拉不到 Docker Hub 时，换成内网镜像仓库：
#   docker build --build-arg PYTHON_IMAGE=<内网仓库>/python:3.11-slim \
#                --build-arg UV_IMAGE=<内网仓库>/astral/uv:0.8.17 -t rodit:<版本> .

ARG PYTHON_IMAGE=python:3.11-slim
ARG UV_IMAGE=astral/uv:0.8.17

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- 构建：只装运行依赖（不装 dev 组）
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# 先装依赖（pyproject.toml / uv.lock 不变时这一层走缓存）；--frozen：严格按 uv.lock，不重新解析
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
# 再装项目本身（非可编辑安装：运行镜像里不需要源码目录）
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------- 运行
FROM ${PYTHON_IMAGE} AS runtime
ARG RODIT_UID=10001
RUN groupadd --system --gid "${RODIT_UID}" rodit \
 && useradd --system --uid "${RODIT_UID}" --gid rodit --no-create-home \
      --home-dir /nonexistent --shell /usr/sbin/nologin rodit \
 && mkdir -p /var/lib/rodit \
 && chown rodit:rodit /var/lib/rodit
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY config/rodit.yaml /app/config/rodit.yaml
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RODIT_CONFIG=/app/config/rodit.yaml \
    RODIT_HOST=0.0.0.0 \
    RODIT_PORT=8000
# 非 root 运行；/var/lib/rodit 只给本地试用放令牌文件（生产挂载在 /run/secrets）
USER rodit
EXPOSE 8000
# 只查存活（/healthz）；就绪（/readyz：数据库、迁移、大模型配置）交给编排系统的就绪探针
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-m", "rodit.ops", "healthcheck"]
ENTRYPOINT ["python", "-m", "rodit.ops"]
CMD ["serve"]

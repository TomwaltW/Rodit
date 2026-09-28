#!/usr/bin/env bash
# 云端会话（Claude Code on the web）启动时安装依赖；本机会话直接跳过。
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"
uv sync --frozen

"""部署文件的静态检查（不需要 Docker）：Dockerfile、.dockerignore、本地试用的 compose、密钥模板、
README 运维说明。重点是"仓库里没有任何密钥"（CLAUDE.md 云端开发硬规则）和镜像的基本约束
（python:3.11-slim、uv sync --frozen、不装 dev 依赖、非 root）。"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "deploy" / "docker-compose.yml"
ENV_TEMPLATE = ROOT / "deploy" / "env.example"
INITDB = ROOT / "deploy" / "postgres" / "initdb" / "10-rodit-roles.sh"

INSTRUCTIONS = {
    "FROM", "RUN", "CMD", "LABEL", "EXPOSE", "ENV", "ADD", "COPY", "ENTRYPOINT", "VOLUME",
    "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
}  # fmt: skip
SECRET_NAME = re.compile(r"(PASSWORD|SECRET|TOKEN_KEY|SALT|API_KEY|DSN)", re.IGNORECASE)


def _instructions() -> list[tuple[str, str]]:
    """Dockerfile → [(指令, 参数)]：合并续行、去掉注释和空行（解析器指令 # syntax 除外）。"""
    out: list[tuple[str, str]] = []
    buffer = ""
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not buffer and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        keyword, _, args = buffer.strip().partition(" ")
        out.append((keyword.upper(), args.strip()))
        buffer = ""
    assert not buffer, "Dockerfile 以续行结尾"
    return out


def test_dockerfile_syntax():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert text.startswith("# syntax=docker/dockerfile:1")  # RUN --mount 需要 BuildKit 前端
    instructions = _instructions()
    unknown = [k for k, _ in instructions if k not in INSTRUCTIONS]
    assert not unknown, unknown
    # ARG 之后第一条非 ARG 指令必须是 FROM
    first = next(k for k, _ in instructions if k != "ARG")
    assert first == "FROM"


def test_dockerfile_is_multi_stage_on_python_311_slim():
    instructions = _instructions()
    args = dict(a.split("=", 1) for k, a in instructions if k == "ARG" and "=" in a)
    assert args["PYTHON_IMAGE"] == "python:3.11-slim"
    stages = [a.split() for k, a in instructions if k == "FROM"]
    names = [s[-1] for s in stages if len(s) == 3 and s[1].upper() == "AS"]
    assert names == ["uv", "builder", "runtime"]
    assert all(s[0] in ("${PYTHON_IMAGE}", "${UV_IMAGE}") for s in stages)
    # 最后一个阶段是运行镜像
    assert stages[-1][-1] == "runtime"


def test_dockerfile_uses_frozen_uv_sync_without_dev_deps():
    runs = [a for k, a in _instructions() if k == "RUN"]
    syncs = [r for r in runs if "uv sync" in r]
    assert len(syncs) == 2
    for cmd in syncs:
        assert "--frozen" in cmd and "--no-dev" in cmd
    assert "--no-editable" in syncs[-1]  # 运行镜像不需要源码
    assert not any("pip install" in r for r in runs)


def test_runtime_stage_runs_as_non_root():
    instructions = _instructions()
    runtime_start = max(i for i, (k, _) in enumerate(instructions) if k == "FROM")
    runtime = instructions[runtime_start:]
    users = [a for k, a in runtime if k == "USER"]
    assert users and users[-1] not in ("root", "0", "0:0")
    entry = next(a for k, a in runtime if k == "ENTRYPOINT")
    assert entry == '["python", "-m", "rodit.ops"]'
    assert any(k == "HEALTHCHECK" for k, _ in runtime)
    # 运行阶段只从构建阶段拿虚拟环境，外加配置文件
    copies = [a for k, a in runtime if k == "COPY"]
    assert copies == [
        "--from=builder /app/.venv /app/.venv",
        "config/rodit.yaml /app/config/rodit.yaml",
    ]


def test_dockerfile_has_no_secrets():
    for keyword, args in _instructions():
        if keyword in ("ENV", "ARG"):
            for token in shlex.split(args):
                name = token.split("=", 1)[0]
                assert not SECRET_NAME.search(name), f"镜像里不能放密钥：{name}"


def test_copy_sources_exist_and_are_not_dockerignored():
    ignored = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for keyword, args in _instructions():
        if keyword != "COPY" or args.startswith("--from"):
            continue
        *sources, _dest = args.split()
        for src in sources:
            assert (ROOT / src).exists(), src
            assert src.split("/")[0] not in ignored, src


def test_dockerignore_keeps_secrets_and_junk_out():
    lines = set(DOCKERIGNORE.read_text(encoding="utf-8").splitlines())
    for pattern in (".git", ".env", ".env.*", "**/.env", ".venv", "tests", "deploy", ".claude"):
        assert pattern in lines, pattern


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_compose_services():
    compose = _compose()
    services = compose["services"]
    assert set(services) == {"db", "init", "app"}
    assert services["db"]["image"] == "postgres:16"
    assert "ports" not in services["db"]  # 数据库不对宿主机开放
    assert services["init"]["command"] == ["local-init"]
    assert services["app"]["command"] == ["serve"]
    assert services["app"]["ports"] == ["127.0.0.1:8000:8000"]  # 只监听本机
    assert services["app"]["depends_on"]["init"]["condition"] == "service_completed_successfully"
    assert services["init"]["depends_on"]["db"]["condition"] == "service_healthy"
    for name in ("init", "app"):
        assert services[name]["env_file"] == [".env"]
        assert services[name]["environment"]["RODIT_ENV"] == "local"
    assert services["app"]["read_only"] is True
    assert services["app"]["cap_drop"] == ["ALL"]


def _walk_env(node: object):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "environment" and isinstance(value, dict):
                yield from value.items()
            else:
                yield from _walk_env(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_env(item)


def test_compose_has_no_literal_secrets():
    """密钥只能是 ${变量} 引用（值在被 .gitignore 忽略的 deploy/.env 里）。"""
    for name, value in _walk_env(_compose()):
        value = str(value)
        if SECRET_NAME.search(name):
            assert "${" in value, f"{name} 必须引用变量"
        for secret_ref in re.findall(r"://[^:/@]+:([^@]+)@", value):
            assert secret_ref.startswith("${"), f"{name} 的连接串里不能写死密码"


def test_env_template_has_placeholders_only():
    names: list[str] = []
    for line in ENV_TEMPLATE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        assert sep and value == "", f"模板里只能有占位符：{name}"
        names.append(name)
    for required in (
        "RODIT_TOKEN_KEY",
        "RODIT_BUYER_SALT",
        "RODIT_DB_OWNER_PASSWORD",
        "RODIT_DB_APP_PASSWORD",
    ):
        assert required in names
    # compose 里 ${VAR:?} 引用的变量都在模板里
    referenced = set(re.findall(r"\$\{([A-Z_]+):\?", COMPOSE.read_text(encoding="utf-8")))
    assert referenced <= set(names), referenced - set(names)


def test_gitignore_covers_local_secrets():
    lines = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert ".env" in lines  # deploy/.env
    assert "*tokens*.json" in lines


def test_initdb_script_creates_non_superuser_login_in_app_role():
    text = INITDB.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh")
    assert "NOSUPERUSER NOBYPASSRLS" in text
    assert "GRANT rodit_app TO rodit_login" in text
    assert ":'app_password'" in text  # 密码经 psql 变量传入，不写在脚本里
    assert INITDB.stat().st_mode & 0o111  # 可执行


@pytest.mark.parametrize(
    "needle",
    [
        "RODIT_TOKEN_KEY",
        "RODIT_BUYER_SALT",
        "RODIT_DATABASE_DSN",
        "RODIT_DB_STATEMENT_TIMEOUT_MS",
        "RODIT_LLM_API_KEY",
        "RODIT_PLATFORM_BINDINGS",
        "RODIT_MODE_C_ONLY",
        "/readyz",
        "/metrics",
        "docker compose -f deploy/docker-compose.yml",
    ],
)
def test_readme_runbook_mentions(needle):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## 部署：一个平台单元", 1)[1].split("\n## ", 1)[0]
    assert needle in section

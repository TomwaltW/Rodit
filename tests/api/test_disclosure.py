"""生成式人工智能公示（方案 §12.3：直接调用已备案大模型的应用要公示所用模型的名称和备案号或
上线编号）：/v1/me 带上配置里的 llm.public_name / llm.filing_no，话术助手页脚显示；生产部署
配置了 llm.require_disclosure 却没填这两项时拒绝启动。名称和备案号都是合成值。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn
import yaml
from api_support import CALLERS, TOKENS, Api, TestClient

from rodit.api import StaticTokenAuthenticator, create_app
from rodit.api.web import ASSET_PREFIX, CONTENT_SECURITY_POLICY
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.core.config import EngineConfig, LLMConfig
from rodit.core.enums import Platform
from rodit.ops.serve import StartupError, build_server, preflight
from rodit.ops.settings import ServerSettings

PUBLIC_NAME = "合成大模型 Synth-7B"
FILING_NO = "Synth-000000000000000"  # 合成的备案号（不是任何真实编号）

NODE = shutil.which("node")
SMOKE = Path(__file__).with_name("web_disclosure_smoke.cjs")
SKIP_EXIT = 77


def _api(llm: LLMConfig) -> Api:
    service = AssistService(build_demo_unit(Platform.TAOBAO, config=EngineConfig(llm=llm)))
    authenticator = StaticTokenAuthenticator({TOKENS[n]: p for n, p in CALLERS.items()})
    return Api(TestClient(create_app(service, authenticator=authenticator)), service)


@pytest.fixture
def disclosed() -> Api:
    return _api(LLMConfig(public_name=PUBLIC_NAME, filing_no=FILING_NO))


# ================================================================ /v1/me


@pytest.mark.parametrize("caller", ["agent_a", "manager_a", "merchant_a", "reviewer_b"])
def test_me_includes_disclosure_when_configured(disclosed: Api, caller: str):
    r = disclosed.get("/v1/me", caller)
    assert r.status_code == 200, r.text
    p = CALLERS[caller]
    assert r.json() == {
        "shop_id": p.shop_id,
        "user_id": p.user_id,
        "role": p.role,
        "platform": "taobao",
        "server_time": disclosed.unit.clock.now().isoformat(),
        "llm": {"public_name": PUBLIC_NAME, "filing_no": FILING_NO},
    }


def test_me_without_disclosure_is_unchanged(api: Api):
    """没配置时不带 llm 字段（和原来的响应一模一样）。"""
    body = api.get("/v1/me", "agent_a").json()
    assert "llm" not in body
    assert set(body) == {"shop_id", "user_id", "role", "platform", "server_time"}


def test_me_with_only_a_name_or_blank_values():
    only_name = _api(LLMConfig(public_name=f"  {PUBLIC_NAME} ", filing_no="   "))
    assert only_name.get("/v1/me", "agent_a").json()["llm"] == {"public_name": PUBLIC_NAME}
    blank = _api(LLMConfig(public_name=" ", filing_no=""))
    assert "llm" not in blank.get("/v1/me", "agent_a").json()


def test_disclosure_never_leaks_llm_secrets_or_endpoint(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", "sk-synthetic-only-disclosure-0000")
    api = _api(
        LLMConfig(
            base_url="https://llm.example.invalid/v1",
            model="internal-model-id",
            public_name=PUBLIC_NAME,
            filing_no=FILING_NO,
        )
    )
    text = api.get("/v1/me", "agent_a").text
    for secret in ("sk-synthetic-only-disclosure-0000", "llm.example.invalid", "internal-model-id"):
        assert secret not in text


def test_me_schema_documents_the_disclosure(api: Api):
    schemas = api.get("/openapi.json").json()["components"]["schemas"]
    assert set(schemas["LLMDisclosureView"]["properties"]) == {"public_name", "filing_no"}
    assert "llm" in schemas["MeView"]["properties"]


# ================================================================ 页面


def test_assist_page_has_a_hidden_disclosure_footer_and_same_csp(disclosed: Api):
    r = disclosed.get("/assist")
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    html = r.text
    assert '<footer id="disclosure"' in html
    footer = html[html.index('<footer id="disclosure"') :]
    footer = footer[: footer.index("</footer>")]
    assert " hidden" in footer.split(">", 1)[0]  # 页面本身不含数据：登录后由脚本填、再显示
    assert 'id="llm-public-name"' in footer and 'id="llm-filing-no"' in footer
    assert PUBLIC_NAME not in html and FILING_NO not in html
    assert "style=" not in footer
    js = disclosed.get(ASSET_PREFIX + "assist.js").text
    assert "showDisclosure(me.llm)" in js  # 登录拿到身份后才填、才显示
    assert "innerHTML" not in js


@contextmanager
def _serve(app) -> Iterator[str]:
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning", lifespan="off", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("测试服务器没有启动")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_disclosure_footer_in_a_real_browser(disclosed: Api):
    assert NODE is not None
    env = dict(os.environ)
    global_modules = Path(NODE).resolve().parent.parent / "lib" / "node_modules"
    env["NODE_PATH"] = os.pathsep.join(p for p in (env.get("NODE_PATH"), str(global_modules)) if p)
    with _serve(disclosed.client.app) as base:
        env.update(
            RODIT_BASE_URL=base,
            RODIT_SMOKE_TOKENS=json.dumps({"agent": TOKENS["agent_a"]}),
            RODIT_DISCLOSURE=json.dumps({"public_name": PUBLIC_NAME, "filing_no": FILING_NO}),
        )
        proc = subprocess.run(
            [NODE, str(SMOKE)], env=env, capture_output=True, text=True, timeout=120
        )
    if proc.returncode == SKIP_EXIT:
        pytest.skip(f"没有 Playwright 或浏览器起不来：{proc.stderr.strip()[:300]}")
    lines = proc.stdout.strip().splitlines()
    result = json.loads(lines[-1]) if lines else {"ok": False, "error": proc.stderr}
    assert proc.returncode == 0 and result["ok"], result.get("error") or proc.stderr
    assert result["steps"] == ["page", "shown", "signed-out"]


# ================================================================ 启动前检查


def _settings(env: str = "production") -> ServerSettings:
    return ServerSettings(env=env, platform=Platform.TAOBAO, mode_c_only=True)  # type: ignore[arg-type]


ENV = {
    "RODIT_DATABASE_DSN": "dbname=rodit_never_connected",
    "RODIT_TOKEN_KEY": "rodit-test-token-key-synthetic-only",
    "RODIT_BUYER_SALT": "salt-synthetic-only",
    "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
    "RODIT_LLM_MODEL": "synthetic-model",
}


@pytest.mark.parametrize(
    ("public_name", "filing_no", "missing"),
    [
        (None, None, ["llm.public_name", "llm.filing_no"]),
        (PUBLIC_NAME, None, ["llm.filing_no"]),
        ("  ", FILING_NO, ["llm.public_name"]),
    ],
)
def test_production_requires_disclosure_when_configured(public_name, filing_no, missing):
    config = EngineConfig(
        llm=LLMConfig(public_name=public_name, filing_no=filing_no, require_disclosure=True)
    )
    with pytest.raises(StartupError, match="require_disclosure") as exc:
        preflight(_settings(), config, ENV)
    for name in missing:
        assert name in str(exc.value)
    assert "§12.3" in str(exc.value)


def test_disclosure_check_passes_or_does_not_apply():
    required = LLMConfig(public_name=PUBLIC_NAME, filing_no=FILING_NO, require_disclosure=True)
    preflight(_settings(), EngineConfig(llm=required), ENV)
    # 没要求时不拦（默认配置 require_disclosure: false）；本地试用也不拦
    preflight(_settings(), EngineConfig(), ENV)
    preflight(_settings("local"), EngineConfig(llm=LLMConfig(require_disclosure=True)), ENV)


def test_serve_refuses_to_start_without_disclosure(monkeypatch, tmp_path):
    """经部署入口（RODIT_CONFIG 指向的 YAML）：生产模式、要求公示、没填 → 拒绝启动。"""
    repo_config = Path(__file__).resolve().parents[2] / "config" / "rodit.yaml"
    config = yaml.safe_load(repo_config.read_text(encoding="utf-8"))
    config["llm"]["require_disclosure"] = True
    path = tmp_path / "rodit.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    for name in ("RODIT_ENV", "RODIT_PLATFORM_BINDINGS", "RODIT_METRICS_PUBLIC"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**ENV, "RODIT_PLATFORM": "taobao", "RODIT_MODE_C_ONLY": "1"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RODIT_CONFIG", str(path))
    monkeypatch.setenv("RODIT_TOKENS_FILE", str(tmp_path / "never-read.json"))
    with pytest.raises(StartupError, match="llm.public_name"):
        build_server(ServerSettings.from_env())

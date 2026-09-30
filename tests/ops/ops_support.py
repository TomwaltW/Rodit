"""rodit.ops 测试的公共部分：合成环境变量、假数据库、"平台云代码"的绑定、指标文本解析。

importlib 模式下测试模块不能 import conftest，所以放在单独的模块里，由 conftest 把本目录加进
sys.path。全部是合成数据；不联网（大模型地址是 .invalid，测试里不会真的调用）。
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from starlette.exceptions import StarletteDeprecationWarning

from rodit.adapters import StandaloneAssistAdapter, TaobaoMockAdapter
from rodit.ops.bindings import BindingContext, PlatformBindings, demo_bindings
from rodit.storage.db import MIGRATIONS
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import build_demo_world

with warnings.catch_warnings():
    # 依赖冻结：开发依赖里是 httpx（不是 httpx2），starlette 导入测试客户端时会发弃用警告
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient  # noqa: F401  (re-export)

SYNTHETIC_TOKEN_KEY = "rodit-test-token-key-synthetic-only"
SYNTHETIC_SALT = "salt-synthetic-only"
SYNTHETIC_LLM_KEY = "sk-synthetic-only-0000000000"

PROD_ENV: dict[str, str] = {
    "RODIT_PLATFORM": "taobao",
    "RODIT_DATABASE_DSN": "dbname=rodit_never_connected",
    "RODIT_TOKEN_KEY": SYNTHETIC_TOKEN_KEY,
    "RODIT_BUYER_SALT": SYNTHETIC_SALT,
    "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
    "RODIT_LLM_MODEL": "synthetic-model",
    "RODIT_LLM_API_KEY": SYNTHETIC_LLM_KEY,
}
"""生产模式的合成环境变量（数据库不会真的连：Database 构造时不建连接）。"""

OPS_ENV_NAMES: tuple[str, ...] = (
    "RODIT_ENV",
    "RODIT_PLATFORM",
    "RODIT_PLATFORM_BINDINGS",
    "RODIT_MODE_C_ONLY",
    "RODIT_CONFIG",
    "RODIT_TOKENS_FILE",
    "RODIT_TOKENS_RELOAD_S",
    "RODIT_METRICS_PUBLIC",
    "RODIT_HOST",
    "RODIT_PORT",
    "RODIT_LOG_LEVEL",
    "RODIT_LOG_FORMAT",
    "RODIT_MIGRATE_ON_START",
    "RODIT_AUTH_EXPIRY_WARN_DAYS",
    "RODIT_AUTH_EXPIRY_CHECK_S",
    "RODIT_READINESS_CACHE_S",
    "RODIT_DATABASE_DSN",
    "RODIT_MIGRATE_DSN",
    "RODIT_TOKEN_KEY",
    "RODIT_BUYER_SALT",
    "RODIT_LLM_BASE_URL",
    "RODIT_LLM_MODEL",
    "RODIT_LLM_API_KEY",
    "RODIT_LLM_TIMEOUT_S",
    "RODIT_DB_STATEMENT_TIMEOUT_MS",
    "RODIT_DB_LOCK_TIMEOUT_MS",
)
"""测试开始前全部清掉，只用测试自己设的值。"""


# ================================================================ 假数据库（就绪检查）


class _Result:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


@dataclass
class FakeReadinessDb:
    """实现 health.SessionDatabase：versions 是 schema_migrations 里的版本（默认全部迁移）；
    enter_error / query_error 模拟连不上 / 查询出错。"""

    versions: Sequence[int] = field(default_factory=lambda: tuple(m.version for m in MIGRATIONS))
    enter_error: BaseException | None = None
    query_error: BaseException | None = None
    sessions: int = 0
    statements: list[str] = field(default_factory=list)

    @contextmanager
    def session(self, shop_id: str | None, **kw: Any) -> Iterator[Any]:
        self.sessions += 1
        if self.enter_error is not None:
            raise self.enter_error
        yield self

    def execute(self, statement: str, *args: Any) -> _Result:
        self.statements.append(statement)
        if self.query_error is not None:
            raise self.query_error
        return _Result([{"version": v} for v in self.versions])


# ================================================================ "平台云代码"的绑定


class CloudTools:
    """模拟平台云代码提供的只读工具：内部用合成演示世界，但不是 rodit.testing 的类
    （生产模式的模拟实现检查只认本仓库的模拟类）。"""

    def __init__(self, inner: InMemoryToolGateway) -> None:
        self._inner = inner
        self.platform = inner.platform

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class CloudAdapter:
    """模拟平台云代码提供的模式 B 适配器（转给模拟适配器处理合成事件）。"""

    def __init__(self, inner: TaobaoMockAdapter) -> None:
        self._inner = inner
        self.platform = inner.platform
        self.access = inner.access

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def cloud_bindings(ctx: BindingContext) -> PlatformBindings:
    world = build_demo_world(ctx.clock.now())
    tools = InMemoryToolGateway(world, ctx.platform)
    adapter = TaobaoMockAdapter(tools, ctx.clock, ctx.hasher, access="B")
    return PlatformBindings(
        tools=CloudTools(tools),
        adapters=[
            CloudAdapter(adapter),
            StandaloneAssistAdapter(ctx.platform, ctx.clock, ctx.hasher),
        ],
        shop_ids=lambda: ["s_1001", "s_2002"],
    )


def cloud_bindings_with_mock_catalog(ctx: BindingContext) -> PlatformBindings:
    """平台工具、适配器是"平台云实现"，商品接口和 OCR 却是本仓库的模拟实现（生产模式必须拒绝）。"""
    from dataclasses import replace

    from rodit.testing.fakes import FakeImageTextExtractor, InMemoryItemCatalog
    from rodit.testing.fixtures import demo_platform_items

    world = build_demo_world(ctx.clock.now())
    return replace(
        cloud_bindings(ctx),
        catalog=InMemoryItemCatalog(demo_platform_items(world), ctx.platform),
        ocr=FakeImageTextExtractor({}),
    )


def mock_bindings(ctx: BindingContext) -> PlatformBindings:
    """直接交出本仓库的模拟实现（生产模式必须拒绝）。"""
    return demo_bindings(ctx)


def not_bindings(ctx: BindingContext) -> object:
    return {"tools": None}


NOT_CALLABLE = "这不是函数"


# ================================================================ 指标文本


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(?:,|$)')

Sample = tuple[str, frozenset[tuple[str, str]]]


def parse_exposition(text: str) -> dict[Sample, float]:
    """Prometheus 文本格式 → {(指标名, 标签集合): 值}。格式不对的行直接断言失败。"""
    out: dict[Sample, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            if line.startswith("#"):
                assert line.startswith(("# HELP ", "# TYPE ")), line
            continue
        m = _SAMPLE.match(line)
        assert m, f"不是合法的样本行：{line!r}"
        name, raw_labels, value = m.groups()
        labels: list[tuple[str, str]] = []
        if raw_labels:
            consumed = 0
            for lm in _LABEL.finditer(raw_labels):
                assert lm.start() == consumed, f"标签格式不对：{raw_labels!r}"
                labels.append((lm.group(1), lm.group(2)))
                consumed = lm.end()
            assert consumed == len(raw_labels), f"标签格式不对：{raw_labels!r}"
        key = (name, frozenset(labels))
        assert key not in out, f"样本重复：{line!r}"
        out[key] = float(value.replace("+Inf", "inf"))
    return out


def sample(samples: Mapping[Sample, float], name: str, **labels: str) -> float | None:
    return samples.get((name, frozenset(labels.items())))


def label_values(samples: Mapping[Sample, float]) -> set[str]:
    return {v for (_, labels) in samples for _, v in labels}

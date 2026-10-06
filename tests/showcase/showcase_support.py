"""演示站测试的公共代码（importlib 模式下测试模块不能 import conftest，放在这里）。

时间固定在合成数据的"现在"（fixtures.DEFAULT_NOW）之后几分钟：演示世界的"现在"取整到分钟，
单元时钟每次请求拨到这个时间，结果可以复现。全部是合成数据、离线替身，不联网。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from fastapi.testclient import TestClient

from rodit.core.config import EngineConfig
from rodit.core.enums import Platform
from rodit.eval.llm_env import client_from_env
from rodit.showcase.app import create_showcase_app
from rodit.showcase.state import ShowcaseHub
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import DEFAULT_NOW

NOW = DEFAULT_NOW + timedelta(minutes=3, seconds=20)
"""测试里的"真实时间"（演示世界的现在会取整到 10:38）。"""

BUYER_REF = re.compile(r"\bb_[0-9a-f]{24}\b")
RAW_BUYER = re.compile(r"(?:tb|dy|pdd)_demo_buyer_\d+")
WRITE_PATHS = re.compile(r"/showcase/api/(?:feedback|aftersales/[^/]+/(?:confirm|negotiate))$")


@dataclass
class Showcase:
    """一个演示站应用 + 测试客户端；responses 记下所有响应正文（查有没有泄露买家标识）。"""

    hub: ShowcaseHub
    client: TestClient
    clock: FakeClock
    responses: list[str] = field(default_factory=list)

    def get(self, path: str, **params: Any):
        r = self.client.get(path, params=params or None)
        self.responses.append(r.text)
        return r

    def post(self, path: str, body: Any = None, **kwargs: Any):
        """写操作（推荐反馈、售后确认、协商）没带 epoch 时带上当前单元的（像刚载入的页面）；
        测过期、缺 epoch 的自己在 body 里给，或直接用 client。"""
        if isinstance(body, dict) and "epoch" not in body and WRITE_PATHS.search(path):
            epoch = self.epoch(body.get("platform"))
            if epoch is not None:
                body = {**body, "epoch": epoch}
        r = self.client.post(path, json=body, **kwargs)
        self.responses.append(r.text)
        return r

    def epoch(self, platform: Any) -> str | None:
        """当前演示单元的 epoch（不认识的平台返回 None）。"""
        try:
            return self.hub.get(Platform(platform)).epoch
        except ValueError:
            return None

    def chat(
        self,
        platform: Platform | str,
        scenario: str,
        text: str | None = None,
        images: list[str] | None = None,
    ) -> dict:
        body: dict[str, Any] = {"platform": str(platform), "scenario": scenario}
        if text is not None:
            body["text"] = text
        if images is not None:
            body["images"] = images
        r = self.post("/showcase/api/chat", body)
        assert r.status_code == 200, r.text
        return r.json()

    def feedback(self, platform: Platform | str, suggestion_id: int, action: str, **extra: Any):
        body = {"platform": str(platform), "suggestion_id": suggestion_id, "action": action}
        body.update(extra)
        return self.post("/showcase/api/feedback", body)

    def dashboard(self, platform: Platform | str) -> dict:
        r = self.get("/showcase/api/dashboard", platform=str(platform))
        assert r.status_code == 200, r.text
        return r.json()

    def aftersales(self, platform: Platform | str) -> dict:
        r = self.get("/showcase/api/aftersales", platform=str(platform))
        assert r.status_code == 200, r.text
        return r.json()

    def aftersale(self, platform: Platform | str, aftersale_id: str) -> dict:
        items = self.aftersales(platform)["items"]
        return next(i for i in items if i["aftersale_id"] == aftersale_id)

    def all_text(self) -> str:
        return "\n".join(self.responses)


def build_showcase(*, history: bool = False) -> Showcase:
    """history=False 时不回放历史（组装只要零点几秒），看板只有演示里的操作。"""
    clock = FakeClock(NOW)
    hub = ShowcaseHub(clock=clock, history_seed=20260928 if history else None)
    # 演示站默认只认本机地址的 Host（防 DNS 重绑定，见 showcase.app）
    client = TestClient(create_showcase_app(hub), base_url="http://127.0.0.1")
    return Showcase(hub=hub, client=client, clock=clock)


def build_real_showcase(
    *,
    judge: str = "llm",
    latency: str = "strict",
    history: bool = False,
    config: EngineConfig | None = None,
) -> Showcase:
    """接"真实大模型"的演示站：RODIT_LLM_* 要先指向本机的模型替身（conftest 的 stub_env），
    客户端是真实的 OpenAICompatibleClient（rodit.eval.llm_env.client_from_env），走真实的 HTTP。"""
    clock = FakeClock(NOW)
    hub = ShowcaseHub(
        clock=clock,
        llm=client_from_env(),
        judge=judge,  # type: ignore[arg-type]
        latency=latency,  # type: ignore[arg-type]
        config=config,
        history_seed=20260928 if history else None,
    )
    client = TestClient(create_showcase_app(hub), base_url="http://127.0.0.1")
    return Showcase(hub=hub, client=client, clock=clock)


def walk(value: Any) -> Iterator[Any]:
    """JSON 里的所有值（递归）。"""
    yield value
    if isinstance(value, dict):
        for v in value.values():
            yield from walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk(v)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


__all__ = [
    "BUYER_REF",
    "NOW",
    "RAW_BUYER",
    "Showcase",
    "build_real_showcase",
    "build_showcase",
    "json_text",
    "walk",
]

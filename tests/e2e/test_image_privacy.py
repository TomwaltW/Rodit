"""图片理解的个人信息最小化（方案 §5.3、§12.3）：物流截图里的手机号（合成数据 img_buyer_logistics_1
的图中文字"联系电话 13812345678"）不出现在大模型提示词、判断层留痕、审计、推荐记录、接口响应、
侧边栏里——视觉模型按约定打了码时如此，实现漏了打码时引擎也再打一次。
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import timedelta
from typing import Any

import pytest
from e2e_support import judge_state, production_like_llm, stored_texts
from starlette.exceptions import StarletteDeprecationWarning

from rodit.api import Principal, StaticTokenAuthenticator, create_app
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.enums import ImageKind, Intent, Platform
from rodit.core.models import ImageFinding, ImageRef
from rodit.testing.fakes import FakeImageUnderstanding
from rodit.testing.fixtures import SHOP_A, build_demo_world

with warnings.catch_warnings():
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

PHONE = "13812345678"
PHONE_ANY = re.compile(
    r"1\D{0,3}3\D{0,3}8\D{0,3}1\D{0,3}2\D{0,3}3\D{0,3}4\D{0,3}5\D{0,3}6\D{0,3}7\D{0,3}8"
)
TOKEN = "test-token-integration-a-0123456789abcdef"


def _leaky_analyze(vision: FakeImageUnderstanding):
    """违反约定、没打码的实现：summary / extracted_text 原样带着手机号。"""

    def analyze(shop_id: str, image: ImageRef) -> ImageFinding:
        vision.calls.append((shop_id, image.ref))
        return ImageFinding(
            ref=image.ref,
            kind=ImageKind.LOGISTICS_SCREENSHOT,
            confidence=0.9,
            summary=f"物流页面截图：显示已签收，派件员电话 {PHONE}",
            extracted_text=f"中通快递 已签收 联系电话 {PHONE}",
        )

    return analyze


def _service(*, leaky: bool = False) -> tuple[AssistService, Any]:
    llm = production_like_llm(
        [("物流", Intent.LOGISTICS_TRACK), ("签收", Intent.NOT_RECEIVED)],
        {"": {"answer": "亲，我帮您查一下物流哦", "evidence_ids": [], "need_clarify": False}},
    )
    unit = build_demo_unit(Platform.TAOBAO, world=build_demo_world(), llm=llm)
    assert isinstance(unit.vision, FakeImageUnderstanding)
    if leaky:  # 引擎和审核助手用的是组装时注入的同一个实例
        unit.vision.analyze = _leaky_analyze(unit.vision)  # type: ignore[method-assign]
    return AssistService(unit), llm


def _buyer_side(request: Any) -> str:
    """提示词里来自买家一侧的部分：判断层的 state、生成提示词的【对话】（系统说明里本来就提到
    "[手机号]""联系电话"这类词，不算）。"""
    if request.purpose in ("route", "qa"):
        return json.dumps(judge_state(request), ensure_ascii=False)
    user = request.messages[-1].content
    return user[user.index("【对话】") : user.index("【要求】")]


def _event(service: AssistService, msg_id: str, *, caption: str = "[图片]", **kw: Any) -> dict:
    clock = service.unit.clock
    clock.set(clock.now() + timedelta(minutes=1))
    raw = taobao_event(caption, msg_id=msg_id, session_id="c_priv", **kw)
    raw.update(
        msg_type="image",
        pic_key="img_buyer_logistics_1",
        pic_url=f"https://img.example.invalid/x.jpg?phone={PHONE}",
        send_time_ms=int(clock.now().timestamp() * 1000),
    )
    return raw


def _assert_clean(label: str, blob: str) -> None:
    assert PHONE not in blob, label
    assert not PHONE_ANY.search(blob), label


@pytest.mark.parametrize("leaky", [False, True], ids=["masked", "leaky"])
def test_logistics_screenshot_phone_never_leaks(leaky: bool):
    service, llm = _service(leaky=leaky)
    unit = service.unit
    vision = unit.vision
    app = create_app(
        service,
        authenticator=StaticTokenAuthenticator(
            {TOKEN: Principal(shop_id=SHOP_A, user_id="tb_push_a", role="integration")}
        ),
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # 1. 只发图（插件上报，经 HTTP 接口）；2. 随图写了字（直接交给应用服务）；3. 接着问一句
    response = client.post(
        "/v1/platform/B/events", json=_event(service, "p1", order_id="o_1003"), headers=headers
    )
    assert response.status_code == 200
    body = response.text
    assert "买家发来 1 张图片：物流 / 订单截图（90%）" in body  # 提醒客服的摘要照常有
    _assert_clean("api response", body)

    second = service.handle_event("B", _event(service, "p2", caption="显示签收了但我没收到"))
    assert second is not None and second.route is not None
    unit.clock.set(unit.clock.now() + timedelta(minutes=1))
    third = service.handle_event("B", taobao_event("那怎么办", msg_id="p3", session_id="c_priv"))
    assert third is not None  # 历史里的图片描述（来自缓存）同样不带手机号
    results = [second, third]

    assert vision.calls and all(ref == "img_buyer_logistics_1" for _, ref in vision.calls)
    # 大模型（路由、质检、生成）收到的全部提示词
    assert llm.calls, "这条链路应该调过大模型"
    for request in llm.calls:
        _assert_clean(f"llm:{request.purpose}", "\n".join(m.content for m in request.messages))
    # 审计、判断层留痕、推荐记录、缺失知识、标注样本
    for source, text in stored_texts(unit, SHOP_A).items():
        _assert_clean(source, text)
    for record in unit.repos.suggestions.list(SHOP_A):
        _assert_clean("suggestion record", record.model_dump_json())
    # 引擎结果、侧边栏
    for result in results:
        if result is not None:
            _assert_clean("engine result", result.model_dump_json())
    adapter = unit.adapters.get(Platform.TAOBAO, "B")
    _assert_clean(
        "sidebar", json.dumps(adapter.sidebar_alerts(SHOP_A, "c_priv"), ensure_ascii=False)
    )
    _assert_clean(
        "sidebar cards",
        json.dumps([c.model_dump(mode="json") for c in adapter.sidebar(SHOP_A, "c_priv")]),
    )
    # 会话记录里没有图片的临时地址（地址里带着手机号）
    for msg in unit.repos.conversations.recent_messages(SHOP_A, "c_priv"):
        _assert_clean("conversation", msg.model_dump_json())
    # 审计里没有图片标识和识别出的文字，只有张数和类型
    audit = json.dumps(
        [e.detail for e in unit.repos.audit.list_events(SHOP_A)], ensure_ascii=False, default=str
    )
    assert "img_buyer_logistics_1" not in audit and "中通" not in audit
    assert '"image_kinds": ["logistics_screenshot"]' in audit


@pytest.mark.parametrize("leaky", [False, True], ids=["masked", "leaky"])
def test_extracted_text_of_logistics_screenshot_never_reaches_prompts(leaky: bool):
    """物流截图的图中文字（面单）不进描述：打过码的 "[手机号]" 也不进提示词（§5.3 只给需要的）。"""
    service, llm = _service(leaky=leaky)
    result = service.handle_event("B", _event(service, "q1", order_id="o_1003"))
    assert result is not None and llm.calls
    for request in llm.calls:
        buyer_side = _buyer_side(request)
        assert "物流 / 订单截图" in buyer_side
        assert "联系电话" not in buyer_side and "中通快递" not in buyer_side
        if not leaky:
            assert "[手机号]" not in buyer_side

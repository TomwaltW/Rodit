"""端到端：显示签收但没收到（方案 §7.3.1、§11.3）。离线演示单元（关键词路由、复述依据的生成、
启发式质检）从接入层进来：第一条推荐是确定性模板——签收细分状态、驿站 / 快递柜取件指引、
还是没找到就在订单页申请售后；本地演示（python -m rodit.app.demo）的 §11.3 也给出驿站信息。"""

from __future__ import annotations

import io
from datetime import timedelta

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, main, taobao_event
from rodit.core.enums import DraftKind, Intent, RouteAction
from rodit.core.models import EngineResult


def say(service: AssistService, text: str, order_id: str, session: str) -> EngineResult:
    service.unit.clock.set(service.unit.clock.now() + timedelta(minutes=5))
    result = service.handle_event(
        "B", taobao_event(text, msg_id=f"m_{session}", order_id=order_id, session_id=session)
    )
    assert isinstance(result, EngineResult)
    return result


def test_station_locker_and_self_signed():
    service = AssistService(build_demo_unit())
    station = say(service, "显示签收了但我根本没收到", "o_1003", "c1")
    locker = say(service, "我没收到货但是显示签收了", "o_1010", "c2")
    self_signed = say(service, "显示签收了但我根本没收到", "o_1004", "c3")
    for result in (station, locker, self_signed):
        assert result.route is not None and result.route.intent == Intent.NOT_RECEIVED
        assert result.route.action == RouteAction.HIGH_RISK_ANSWER
        first = result.suggestions[0]
        assert first.kind == DraftKind.TEMPLATE
        assert "申请售后" in first.text and "截至" in first.text
        assert "请核查物流签收细分状态" in first.risk_hints  # 高风险类型的风险提示照旧
    assert "菜鸟驿站（幸福里店）" in station.suggestions[0].text
    assert "丰巢" in locker.suggestions[0].text
    assert "本人签收" in self_signed.suggestions[0].text
    assert "驿站" not in self_signed.suggestions[0].text  # 不套别的订单的取件点
    cited = [e.id for e in station.suggestions[0].evidence]
    assert cited == ["tool:logistics:o_1003", "platform:taobao:apply_aftersale"]


def test_demo_shows_the_station_for_11_3():
    out = io.StringIO()
    assert main(out) == 0
    text = out.getvalue()
    block = text.split("=== §11.3 显示签收但没收到", 1)[1].split("\n=== ", 1)[0]
    first = next(line for line in block.splitlines() if line.startswith("推荐 1"))
    assert "［template/" in first
    assert "菜鸟驿站（幸福里店）" in first and "请凭取件码取件" in first and "申请售后" in first

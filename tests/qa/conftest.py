"""质检测试的公共夹具：演示世界 + 按 core.evidence 约定构造证据（代替检索 / 工具模块的产出）。"""

from __future__ import annotations

from datetime import datetime

import pytest

from rodit.core import evidence as E
from rodit.core.enums import (
    AnomalyKind,
    EvidenceSource,
    Intent,
    Platform,
    RouteAction,
)
from rodit.core.models import (
    Evidence,
    LogisticsAnalysis,
    LogisticsAnomaly,
    RefundProgress,
    RouteDecision,
    RoutePart,
)
from rodit.testing.fixtures import SHOP_A, DemoWorld, build_demo_world


class EvidenceFactory:
    """用演示世界的数据构造各类证据，data 结构与 core.evidence 模块文档一致。"""

    def __init__(self, world: DemoWorld) -> None:
        self.world = world

    def policy(self, key: str, shop_id: str = SHOP_A) -> Evidence:
        value = self.world.policy_map(shop_id)[key].value
        return Evidence(
            id=E.policy(key),
            source=EvidenceSource.POLICY,
            label=E.policy_label(key),
            content=value,
            data={"key": key, "value": value},
            topic=E.topic_for_policy(key),
        )

    def faq(self, question: str, score: float, *, faq_id: int = 1, shop_id: str = SHOP_A):
        item = next(f for f in self.world.faqs if f.shop_id == shop_id and f.question == question)
        return Evidence(
            id=E.faq(faq_id),
            source=EvidenceSource.FAQ,
            label=E.faq_label(question),
            content=item.answer,
            data={"question": question, "answer": item.answer, "faq_id": faq_id},
            score=score,
        )

    def item_attr(self, name: str, item_id: str = "i_555", shop_id: str = SHOP_A) -> Evidence:
        value = self.world.item(shop_id, item_id).attributes[name]
        return Evidence(
            id=E.item_attr(item_id, name),
            source=EvidenceSource.ITEM,
            label=E.item_label(name),
            content=f"{name}：{value}",
            data={"name": name, "value": value},
        )

    def size_advice(self, item_id: str = "i_555", shop_id: str = SHOP_A) -> Evidence:
        text = self.world.item(shop_id, item_id).size_advice or ""
        return Evidence(
            id=E.item_size_advice(item_id),
            source=EvidenceSource.ITEM,
            label=E.item_label("尺码建议"),
            content=text,
            data={"text": text},
        )

    def platform_rule(self, key: E.PlatformRuleKey, text: str, platform=Platform.TAOBAO):
        return Evidence(
            id=E.platform_rule(platform, key),
            source=EvidenceSource.PLATFORM,
            label=E.platform_rule_label(platform, key),
            content=text,
            data={"platform": str(platform), "key": str(key), "text": text},
        )

    def logistics(
        self,
        order_id: str,
        *,
        summary: str,
        pickup: str | None = None,
        anomalies: list[LogisticsAnomaly] | None = None,
        eta: datetime | None = None,
        as_of: datetime | None = None,
        content: str | None = None,
        shop_id: str = SHOP_A,
    ) -> tuple[Evidence, LogisticsAnalysis]:
        trace = self.world.trace(shop_id, order_id)
        analysis = LogisticsAnalysis(
            status=trace.status,
            as_of=as_of or trace.last_event.time,
            summary=summary,
            anomalies=anomalies or [],
            pickup_guidance=pickup,
            eta=eta,
        )
        ev = Evidence(
            id=E.tool_logistics(order_id),
            source=EvidenceSource.REALTIME,
            label=E.tool_label("logistics"),
            content=content or summary,
            data={
                "trace": trace.model_dump(mode="json"),
                "analysis": analysis.model_dump(mode="json"),
            },
            fetched_at=trace.fetched_at,
        )
        return ev, analysis

    def o_1002(self) -> Evidence:
        """§11.2：运输中，截至 10:30 已到达杭州转运中心。"""
        return self.logistics("o_1002", summary="运输中，已到达杭州转运中心")[0]

    def o_1003(self) -> Evidence:
        """§11.3：菜鸟驿站代收，给取件指引。"""
        return self.logistics(
            "o_1003",
            summary="已签收，菜鸟驿站代收",
            pickup="包裹放在菜鸟驿站（幸福里店），请带上取件码到驿站领取",
            anomalies=[
                LogisticsAnomaly(
                    kind=AnomalyKind.STATION_OR_LOCKER,
                    detail="驿站代收",
                    agent_alert="驿站签收，买家可能没看到取件通知",
                )
            ],
        )[0]

    def o_1006_stalled(self) -> Evidence:
        """§7.2：超过 48 小时没有新轨迹。"""
        return self.logistics(
            "o_1006",
            summary="运输中，已到达武汉转运中心",
            anomalies=[
                LogisticsAnomaly(
                    kind=AnomalyKind.STALLED,
                    detail="超过 48 小时没有新轨迹",
                    agent_alert="建议向快递催查",
                    buyer_hint="已经帮您联系快递催查，有新进展会及时告诉您",
                )
            ],
        )[0]

    def refund(
        self, aftersale_id: str, summary: str, shop_id: str = SHOP_A
    ) -> tuple[Evidence, RefundProgress]:
        info = self.world.aftersale(shop_id, aftersale_id)
        progress = RefundProgress(
            aftersale_id=aftersale_id, status=info.status, summary=summary, as_of=info.updated_at
        )
        ev = Evidence(
            id=E.tool_aftersale(aftersale_id),
            source=EvidenceSource.REALTIME,
            label=E.tool_label("aftersale"),
            content=summary,
            data={
                "aftersale": info.model_dump(mode="json"),
                "progress": progress.model_dump(mode="json"),
            },
            fetched_at=self.world.now,
        )
        return ev, progress


def route_for(
    intent: Intent,
    action: RouteAction = RouteAction.ANSWER,
    *,
    rule: int = 6,
    top: list[tuple[Intent, float]] | None = None,
    parts: list[RoutePart] | None = None,
    high_risk: bool = False,
    risk_hints: list[str] | None = None,
) -> RouteDecision:
    return RouteDecision(
        action=action,
        rule=rule,
        intent=intent,
        intent_confidence=0.9,
        top_intents=top or [(intent, 0.9)],
        parts=parts or [],
        high_risk=high_risk,
        risk_hints=risk_hints or [],
        provider="fake",
    )


@pytest.fixture
def world() -> DemoWorld:
    return build_demo_world()


@pytest.fixture
def ev(world: DemoWorld) -> EvidenceFactory:
    return EvidenceFactory(world)


@pytest.fixture
def route():
    return route_for

"""演示站自检：python -m rodit.showcase --selfcheck [--platform …|all] [--llm env] [--judge …]
[--latency …] [--json OUT]

把每个平台的每个预设场景（showcase.scenarios）都在演示站里发一遍（和页面上点场景走同一条路：
ShowcaseHub → PlatformDemo.chat，不起 HTTP 服务），逐个打印：路由的问题类型、置信度、命中的
§6.6 规则、动作、实际回答的判断层（大模型投票 / 关键词规则兜底）和票数；推荐的类型、是否填进
输入框；质检拦了几份、过了几份；降级；各环节耗时；大模型调用次数和 token。最后汇总：
各环节 P50 / P95 对照 §12.1 目标、降级次数、检查通过情况。

检查分两种：
- 必查（MUST）：不管模型怎么措辞，一个正确的系统都必须做到的（站外交易 → 规则 1 平台标准话术；
  投诉 → 不推荐、提醒人工；§11.5 的模拟承诺被拦下、重写稿通过质检才推荐；分条时两条都有推荐；
  尺码问题用上了商家的尺码建议；质检拦下的稿子不推荐出去；页面数据里没有买家标识和手机号；
  判断层降级时高风险类型不推荐（§12.2）；回放历史不调用真实大模型……）。
  有前提的必查（"按多问题分条时……"）前提不成立时记为"不适用"并写明原因，不算失败；
- 参考（INFO）：和模型、时延有关、不算错的（问题类型和场景标注一致、端到端在 5 秒内……）。

退出码：0 必查全部通过；1 有必查没通过；2 配置错误（由 __main__ 处理）。
报告里没有密钥；大模型接口只写主机名。
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rodit.core.enums import Platform
from rodit.eval.perf import STAGE_LABELS, latency_report
from rodit.routing.policy import ALERT_EMOTION
from rodit.showcase.llm import SCRIPTED_PROMISE
from rodit.showcase.realchain import ChainInfo
from rodit.showcase.scenarios import Scenario, scenarios_for
from rodit.showcase.state import PlatformDemo, ShowcaseHub
from rodit.showcase.trace import PURPOSE_LABELS

Status = Literal["pass", "fail", "skip"]

BUYER_REF = re.compile(r"\bb_[0-9a-f]{24}\b")
RAW_BUYER = re.compile(r"(?:tb|dy|pdd)_demo_buyer_\d+")
PHONE = re.compile(r"(?<![0-9A-Za-z])1[3-9]\d{9}(?![0-9A-Za-z])")
"""手机号（前后不能是字母数字：演示单元的 epoch 是十六进制串，里面的一串数字不算）。"""
OFF_PLATFORM_SCRIPT = "平台规定不能通过站外联系和交易"
PRICE_HINT = "涉及价格承诺，请确认"
DETAIL_SHIP_HOURS = {Platform.TAOBAO: "24小时", Platform.DOUYIN: "24小时", Platform.PDD: "12小时"}
"""合成数据里详情页写的发货时效（和店铺政策冲突，应被覆盖，§5.2.1）。"""

PLATFORM_NAMES = {Platform.TAOBAO: "淘宝", Platform.DOUYIN: "抖店", Platform.PDD: "拼多多"}
STATUS_MARKS = {"pass": "✓", "fail": "✗", "skip": "—"}
OVERRIDE_LABEL = "详情页的发货时效被店铺政策覆盖（§5.2.1）"


@dataclass
class Check:
    id: str
    label: str
    must: bool
    status: Status
    detail: str | None = None


@dataclass
class ScenarioResult:
    platform: str
    scenario: str
    title: str
    section: str
    route: dict[str, Any] | None
    suggestions: list[dict[str, Any]]
    prefill: int | None
    qa: dict[str, Any]
    degraded: list[str]
    latency_ms: dict[str, int]
    llm: dict[str, Any] | None
    checks: list[Check] = field(default_factory=list)
    error: str | None = None
    wall_ms: int = 0


class _Checks:
    """收集一个场景的检查。"""

    def __init__(self) -> None:
        self.items: list[Check] = []

    def must(self, cid: str, label: str, ok: bool, detail: str | None = None) -> None:
        status: Status = "pass" if ok else "fail"
        self.items.append(Check(cid, label, True, status, None if ok else detail))

    def info(self, cid: str, label: str, ok: bool, detail: str | None = None) -> None:
        status: Status = "pass" if ok else "fail"
        self.items.append(Check(cid, label, False, status, None if ok else detail))

    def skip(self, cid: str, label: str, reason: str, *, must: bool = True) -> None:
        self.items.append(Check(cid, label, must, "skip", reason))


def _routed_as(u: dict[str, Any]) -> str:
    return f"路由成了「{u.get('intent_label')}」"


def _skip_reason(u: dict[str, Any], degraded: bool) -> str:
    return "判断层降级（高风险类型不推荐）" if degraded else _routed_as(u)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _evidence_ids(trace: dict[str, Any], intent: str | None = None) -> set[str]:
    return {
        e["id"]
        for p in trace["parts"]
        if p["find"] and (intent is None or p["intent"] == intent)
        for e in p["find"]["evidence"]
    }


def _llm_drafts(trace: dict[str, Any]) -> list[dict[str, Any]]:
    drafts = [d for p in trace["parts"] for a in p["write"] for d in a["drafts"]]
    return [d for d in drafts if d["kind"] == "llm"]


def common_checks(
    c: _Checks, view: dict[str, Any], secrets: Sequence[str], expected_intent: str
) -> None:
    """每个场景都要做到的必查（另加两项参考）。"""
    result, trace = view["result"], view["trace"]
    u = trace["understand"]
    cards = result["suggestions"]
    texts = [card["text"] for card in cards]
    checks = [check for p in trace["parts"] for check in p["judge"]]
    c.must("routed", "消息走到了路由", u is not None, "没有路由结果")
    blocked = [check["text"] for check in checks if not check["passed"]]
    leaked = [t for t in blocked if t in texts]
    c.must(
        "blocked_not_shown",
        "质检拦下的稿子没有推荐出去",
        not leaked,
        f"被拦下的稿子出现在推荐里：{leaked[:1]}",
    )
    c.must(
        "no_scripted_promise",
        "模拟的没依据承诺没有推荐出去（§11.5）",
        SCRIPTED_PROMISE not in texts,
        "推荐里有「明天一定能到」",
    )
    payload = json.dumps(view, ensure_ascii=False)
    found = [p.pattern for p in (BUYER_REF, RAW_BUYER, PHONE) if p.search(payload)]
    found += ["密钥"] * any(s and s in payload for s in secrets)
    c.must(
        "no_buyer_data",
        "页面数据里没有买家标识、手机号、密钥（§5.3）",
        not found,
        f"出现了：{'、'.join(found)}",
    )
    if u is not None and u["degraded"] and u["high_risk"]:
        answered = [card for card in cards if card["kind"] != "template"]
        c.must(
            "degraded_high_risk",
            "判断层降级时高风险类型不推荐回答（§12.2）",
            not answered,
            f"降级时给了 {len(answered)} 条回答",
        )
    if u is not None:
        c.info(
            "intent_matches",
            "问题类型和场景标注一致",
            u["intent"] == expected_intent,
            f"路由成了「{u['intent_label']}」",
        )
    e2e = trace["latency"]["since_received_ms"]
    c.info("e2e", "端到端 ≤ 5 秒（§12.1）", e2e <= trace["latency"]["target_ms"], f"{e2e} ms")


def scenario_checks(c: _Checks, platform: Platform, s: Scenario, view: dict[str, Any]) -> None:
    """每个场景自己的检查（见模块说明）。"""
    result, trace = view["result"], view["trace"]
    u = trace["understand"] or {}
    cards = result["suggestions"]
    first = cards[0] if cards else None
    intent = u.get("intent")
    degraded = bool(u.get("degraded"))
    sid = s.id

    if sid in ("size", "photo_size"):
        if intent != "size_advice":
            c.skip("size_evidence", "检索到商家填的尺码建议", _routed_as(u))
        else:
            ids = _evidence_ids(trace)
            c.must(
                "size_evidence",
                "检索到商家填的尺码建议",
                any(i.endswith(":size_advice") for i in ids),
                "依据里没有尺码建议",
            )
            if cards:
                cited = {e["id"] for card in cards for e in card["evidence"]}
                c.must(
                    "size_cited",
                    "推荐引用了尺码依据",
                    any(":size_" in i for i in cited),
                    f"推荐引用的是 {sorted(cited)}",
                )
            c.info("size_m", "推荐 M 码", any("M" in card["text"] for card in cards))

    elif sid == "ship_time":
        detail = DETAIL_SHIP_HOURS[platform]
        finds = [p["find"] for p in trace["parts"] if p["find"]]
        if not finds:
            c.skip("detail_overridden", OVERRIDE_LABEL, "这次没有检索依据")
        else:
            overridden = [o for f in finds for o in f["overridden"]]
            c.must(
                "detail_overridden",
                OVERRIDE_LABEL,
                any(detail in _squash(o.get("content") or "") for o in overridden),
                "详情页那句没有被覆盖",
            )
        ship_cards = [card for card in cards if card["intent"] == "shipping_policy"]
        if ship_cards:
            c.must(
                "policy_wins",
                "发货时效按店铺政策回答，没有用详情页的说法",
                not any(detail in _squash(card["text"]) for card in ship_cards),
                f"推荐里出现了详情页的「{detail}」",
            )
        c.info("split", "按两个问题分条（§6.6 规则 5）", u.get("action") == "split_multi")

    elif sid in ("logistics", "photo_logistics"):
        if intent != "logistics_track":
            c.skip("logistics_template", "物流用模板回答、写明「截至」", _routed_as(u))
        else:
            c.must(
                "logistics_template",
                "物流用模板回答、写明「截至」（§11.2）",
                first is not None and first["kind"] == "template" and "截至" in first["text"],
                "第一条不是带「截至」的物流模板",
            )
            c.must(
                "logistics_no_llm",
                "物流问题没有调用大模型生成",
                not _llm_drafts(trace),
                "生成了大模型草稿",
            )

    elif sid == "not_received":
        if intent != "not_received" or degraded:
            c.skip(
                "pickup_guidance",
                "签收未收到：驿站取件指引 + 引导申请售后 + 风险提示（§11.3）",
                _skip_reason(u, degraded),
            )
        else:
            c.must(
                "pickup_guidance",
                "签收未收到：驿站取件指引 + 引导申请售后 + 风险提示（§11.3）",
                first is not None
                and first["kind"] == "template"
                and "驿站" in first["text"]
                and "申请售后" in first["text"]
                and bool(first["risk_hints"]),
                "第一条不是带取件指引和风险提示的模板",
            )

    elif sid == "off_platform":
        c.must(
            "rule1",
            "站外交易命中 §6.6 规则 1",
            u.get("rule") == 1 and u.get("action") == "off_platform_script",
            f"命中的是规则 {u.get('rule')}（{u.get('action')}）",
        )
        c.must(
            "platform_script",
            "只推荐审核过的平台规则标准话术、不调用大模型（§11.4）",
            len(cards) == 1
            and cards[0]["kind"] == "template"
            and OFF_PLATFORM_SCRIPT in cards[0]["text"]
            and not _llm_drafts(trace),
            "推荐不是平台标准话术",
        )

    elif sid == "qa_block":
        if not trace.get("simulated_note"):
            c.skip(
                "first_blocked",
                "模拟的第一稿被质检拦下（§11.5）",
                f"这次没有走大模型生成（路由成了「{u.get('intent_label')}」），没有模拟第一稿",
            )
        else:
            checks = [check for p in trace["parts"] for check in p["judge"]]
            first_check = next((ch for ch in checks if ch["text"] == SCRIPTED_PROMISE), None)
            c.must(
                "first_blocked",
                "模拟的第一稿被质检拦下（§11.5）",
                first_check is not None and not first_check["passed"],
                "第一稿没有被拦下",
            )
            rewrites = [ch for ch in checks if ch["attempt"] > 1]
            unavailable = [ch for ch in rewrites if ch["degraded"] or not ch["model_checked"]]
            if not rewrites:
                c.must("rewrite_passed", "带着原因重写、重写稿通过质检后推荐", False, "没有重写")
            elif unavailable and not any(ch["passed"] for ch in rewrites):
                c.skip(
                    "rewrite_passed",
                    "带着原因重写、重写稿通过质检后推荐",
                    "重写稿的模型质检不可用或时间不够（降级），没有推荐大模型回答",
                )
            else:
                passed = [ch for ch in rewrites if ch["passed"]]
                c.must(
                    "rewrite_passed",
                    "带着原因重写、重写稿通过质检后推荐",
                    bool(passed) and any(card["kind"] == "llm" for card in cards),
                    "重写稿没有通过质检",
                )

    elif sid == "emotion":
        c.must(
            "soothe",
            "情绪激动：§6.6 规则 3，先推荐安抚话术并标红",
            u.get("rule") == 3
            and first is not None
            and first["kind"] == "template"
            and "抱歉" in first["text"]
            and ALERT_EMOTION in result["alerts"],
            f"命中的是规则 {u.get('rule')}",
        )

    elif sid == "complaint":
        c.must(
            "human_only",
            "投诉：不推荐、提醒客服优先处理（§5.7）",
            not cards and bool(result["alerts"]) and trace["fallback"]["human_only"],
            f"给了 {len(cards)} 条推荐" if cards else "没有提醒人工",
        )

    elif sid == "multi":
        if u.get("action") != "split_multi":
            c.skip("two_parts", "按问题分条时两条都有推荐", "这次没有按多问题分条")
            c.info("split", "按两个问题分条（§6.6 规则 5）", False, "没有分条")
        else:
            part_intents = [p["intent"] for p in u.get("parts", [])]
            card_intents = {card["intent"] for card in cards}
            c.must(
                "two_parts",
                "按问题分条时两条都有推荐",
                len(set(part_intents)) >= 2 and set(part_intents) <= card_intents,
                f"分条 {part_intents}，推荐 {sorted(card_intents)}",
            )
            c.info(
                "split",
                "分成「商品参数」「发票」两条",
                set(part_intents) == {"product_spec", "invoice_gift"},
            )

    elif sid == "invoice":
        if intent != "invoice_gift":
            c.skip("faq_direct", "FAQ 直出商家审核过的标准回答（§5.4）", _routed_as(u))
        else:
            c.must(
                "faq_direct",
                "FAQ 直出商家审核过的标准回答（§5.4）",
                first is not None and first["kind"] == "faq" and "电子普通发票" in first["text"],
                "第一条不是 FAQ 标准回答",
            )

    elif sid == "ship_quick":
        if intent != "shipping_policy" or degraded:
            c.skip(
                "semi_auto",
                "半自动：高置信度推荐填进输入框 / 标「可直接发送」（§5.6）",
                "判断层降级" if degraded else _routed_as(u),
            )
        elif u.get("confidence", 0) < 0.9:
            c.skip(
                "semi_auto",
                "半自动：高置信度推荐填进输入框 / 标「可直接发送」（§5.6）",
                f"置信度 {u.get('confidence', 0):.0%} 没到半自动阈值 90%",
            )
        else:
            ok = (
                result["input_prefill"] is not None
                if result["prefill_mode"] == "input"
                else any(card["prefill"] for card in cards)
            )
            c.must(
                "semi_auto",
                "半自动：高置信度推荐填进输入框 / 标「可直接发送」（§5.6）",
                ok,
                "没有填进输入框",
            )

    elif sid == "price":
        if intent != "price_promo" or degraded:
            c.skip(
                "price_hint",
                "价格类推荐带「涉及价格承诺，请确认」（§6.6 规则 4）",
                _skip_reason(u, degraded),
            )
        else:
            c.must(
                "price_hint",
                "价格类推荐带「涉及价格承诺，请确认」（§6.6 规则 4）",
                any(PRICE_HINT in card["risk_hints"] for card in cards) or not cards,
                "推荐没有价格风险提示",
            )
            c.info(
                "price_text",
                "只说活动配置里的「满 199 减 20」",
                any("199" in card["text"] for card in cards),
            )

    elif sid == "photo_damage":
        images = trace.get("images") or {}
        c.must(
            "image_seen",
            "买家图片经过图片理解（合成数据的替身）",
            images.get("total") == 1 and not images.get("degraded"),
            "图片没有识别",
        )
        if intent == "quality_issue" and not degraded:
            c.must(
                "photo_template",
                "破损照片：第一条是「照片看到了，请在订单页申请售后」模板，不承诺处理结果",
                first is not None and first["kind"] == "template" and "申请售后" in first["text"],
                "第一条不是照片模板",
            )
        else:
            c.skip(
                "photo_template",
                "破损照片：第一条是「照片看到了，请在订单页申请售后」模板，不承诺处理结果",
                "判断层降级" if degraded else _routed_as(u),
            )


def _route_view(view: dict[str, Any]) -> dict[str, Any] | None:
    u = view["trace"]["understand"]
    if u is None:
        return None
    judge = u.get("judge")
    votes = None
    if judge:
        questions = {q["key"]: q for q in judge["questions"]}

        def counts(key: str) -> list[dict[str, Any]]:
            return list(questions.get(key, {}).get("counts", []))

        votes = {
            "samples": judge["samples"],
            "answered": judge["answered"],
            "errors": judge["errors"],
            "intent": counts("intent"),
            "second_intent": counts("second_intent"),
            "probability_source": judge["probability_source"],
        }
    return {
        "intent": u["intent"],
        "intent_label": u["intent_label"],
        "confidence": u["confidence"],
        "rule": u["rule"],
        "action": u["action"],
        "action_label": u["action_label"],
        "provider": u["provider"],
        "provider_label": u["provider_label"],
        "degraded": u["degraded"],
        "fallback": u.get("fallback"),
        "votes": votes,
    }


def run_scenario(demo: PlatformDemo, s: Scenario, secrets: Sequence[str]) -> ScenarioResult:
    platform = demo.platform
    started = time.perf_counter()
    try:
        view = demo.chat(s.id)
    except Exception as exc:  # noqa: BLE001 — 自检要把异常记成失败，接着跑别的场景
        res = ScenarioResult(
            platform.value, s.id, s.title, s.section, None, [], None, {}, [], {}, None
        )
        res.error = f"{type(exc).__name__}: {exc}"[:200]
        res.checks.append(Check("ran", "场景跑完", True, "fail", res.error))
        return res
    wall = int((time.perf_counter() - started) * 1000)
    trace, result = view["trace"], view["result"]
    checks = _Checks()
    try:
        common_checks(checks, view, secrets, s.intent.value)
        scenario_checks(checks, platform, s, view)
    except (KeyError, TypeError, IndexError) as exc:  # 结果的结构不对也记成失败
        checks.must("checked", "检查跑完", False, f"{type(exc).__name__}: {exc}"[:200])
    lat = {st["key"]: st["ms"] for st in trace["latency"]["stages"]}
    lat["total"] = trace["latency"]["since_received_ms"]
    qa_checks = [ch for p in trace["parts"] for ch in p["judge"]]
    return ScenarioResult(
        platform=platform.value,
        scenario=s.id,
        title=s.title,
        section=s.section,
        route=_route_view(view),
        suggestions=[
            {"kind": card["kind"], "kind_label": card["kind_label"], "intent": card["intent"]}
            for card in result["suggestions"]
        ],
        prefill=result["input_prefill"],
        qa={
            "checked": len(qa_checks),
            "blocked": sum(1 for ch in qa_checks if not ch["passed"]),
            "passed": sum(1 for ch in qa_checks if ch["passed"]),
            "model_checked": sum(1 for ch in qa_checks if ch["model_checked"]),
            "unavailable": sum(1 for ch in qa_checks if ch["degraded"]),
            "out_of_time": sum(
                1 for ch in qa_checks if any("时间不够" in r for r in ch["reasons"])
            ),
        },
        degraded=[d["label"] for d in trace["fallback"]["degraded"]],
        latency_ms=lat,
        llm=trace.get("llm"),
        checks=checks.items,
        wall_ms=wall,
    )


def _platform_checks(demo: PlatformDemo, chain: ChainInfo) -> list[Check]:
    if not chain.real:
        return []
    ok = demo.replay_llm_calls == 0
    return [
        Check(
            "replay_offline",
            f"回放历史 {demo.history.messages} 条消息，没有调用真实大模型",
            True,
            "pass" if ok else "fail",
            None if ok else f"回放期间调用了 {demo.replay_llm_calls} 次",
        )
    ]


def summarize(
    results: Sequence[ScenarioResult], platform_checks: Sequence[Check]
) -> dict[str, Any]:
    ran = [r for r in results if r.error is None]
    lat = latency_report([r.latency_ms for r in ran])
    stages = {
        key: {
            "label": STAGE_LABELS[key],
            "p50": st.p50,
            "p95": st.p95,
            "target_ms": st.target_ms,
            "within_target": st.within_target,
        }
        for key, st in lat.stages.items()
    }
    all_checks = [*platform_checks, *(c for r in results for c in r.checks)]
    must = [c for c in all_checks if c.must]
    info = [c for c in all_checks if not c.must]
    calls: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    tokens = {"input": 0, "output": 0}
    for r in ran:
        for row in (r.llm or {}).get("by_purpose", []):
            calls[row["purpose"]] += row["calls"]
            failed[row["purpose"]] += row["failed"]
            tokens["input"] += row["input_tokens"]
            tokens["output"] += row["output_tokens"]
    route_fallbacks = sum(1 for r in ran if r.route and r.route["degraded"])
    return {
        "scenarios": len(results),
        "errors": sum(1 for r in results if r.error),
        "latency": stages,
        "e2e_p95_within_target": lat.e2e_p95_within_target,
        "degradations": {
            "route_fallback": route_fallbacks,
            "qa_unavailable": sum(r.qa.get("unavailable", 0) for r in ran),
            "qa_out_of_time": sum(r.qa.get("out_of_time", 0) for r in ran),
            "labels": dict(Counter(label for r in ran for label in r.degraded)),
        },
        "llm": {
            "calls": dict(calls),
            "failed": dict(failed),
            "input_tokens": tokens["input"],
            "output_tokens": tokens["output"],
        },
        "must": {
            "total": len(must),
            "passed": sum(1 for c in must if c.status == "pass"),
            "failed": sum(1 for c in must if c.status == "fail"),
            "skipped": sum(1 for c in must if c.status == "skip"),
        },
        "info": {
            "total": len(info),
            "passed": sum(1 for c in info if c.status == "pass"),
        },
        "ok": all(c.status != "fail" for c in must),
    }


def run_selfcheck(
    hub: ShowcaseHub,
    platforms: Sequence[Platform],
    *,
    secrets: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """跑一遍（见模块说明），返回报告（JSON 可序列化）。secrets 是不能出现在页面数据里的值
    （密钥），只用来核对，不写进报告。"""
    results: list[ScenarioResult] = []
    platform_checks: list[Check] = []
    builds: dict[str, Any] = {}
    for platform in platforms:
        if progress:
            progress(f"组装{PLATFORM_NAMES[platform]}演示单元（回放合成历史）…")
        demo = hub.get(platform)
        builds[platform.value] = {
            "history_messages": demo.history.messages,
            "build_ms": demo.build_ms,
            "replay_llm_calls": demo.replay_llm_calls,
        }
        platform_checks.extend(_platform_checks(demo, hub.chain))
        for s in scenarios_for(platform):
            res = run_scenario(demo, s, secrets)
            results.append(res)
            if progress:
                progress(render_scenario(res))
    return {
        "chain": hub.chain.view(),
        "platforms": builds,
        "platform_checks": [asdict(c) for c in platform_checks],
        "scenarios": [asdict(r) for r in results],
        "summary": summarize(results, platform_checks),
    }


# ================================================================ 文本报告


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:.2f} 秒" if value >= 1000 else f"{int(value)} ms"


def render_scenario(r: ScenarioResult | dict[str, Any]) -> str:
    d = asdict(r) if isinstance(r, ScenarioResult) else r
    platform = PLATFORM_NAMES[Platform(d["platform"])]
    lines = [f"[{platform}] {d['scenario']} · {d['title']}（{d['section']}）"]
    if d["error"]:
        lines.append(f"  ✗ 出错：{d['error']}")
        return "\n".join(lines)
    route = d["route"]
    if route:
        votes = ""
        if route["votes"]:
            v = route["votes"]
            counts = "、".join(f"{c['label']} {c['votes']} 票" for c in v["intent"])
            seconds = "、".join(
                f"{c['label']} {c['votes']} 票" for c in v["second_intent"] if c["option"] != "无"
            )
            if seconds:
                counts += f"；另一个问题：{seconds}"
            errs = "、".join(f"{e['label']} ×{e['count']}" for e in v["errors"])
            votes = f" · 投票 {v['answered']}/{v['samples']}" + (f"（{counts}）" if counts else "")
            votes += f"，失败：{errs}" if errs else ""
        lines.append(
            f"  路由：{route['intent_label']} 置信度 {route['confidence']:.0%}"
            f" · 规则 {route['rule']} · {route['action_label']}"
            f" · {route['provider_label']}{votes}"
        )
        if route["fallback"]:
            lines.append(f"  ! {route['fallback']}")
    kinds = "、".join(s["kind_label"] for s in d["suggestions"]) or "无"
    prefill = "是" if d["prefill"] is not None else "否"
    lines.append(f"  推荐：{kinds} · 填进输入框：{prefill}")
    qa = d["qa"]
    lines.append(
        f"  质检：{qa.get('checked', 0)} 份，拦下 {qa.get('blocked', 0)}，"
        f"通过 {qa.get('passed', 0)}（模型质检 {qa.get('model_checked', 0)}，"
        f"不可用 {qa.get('unavailable', 0)}，时间不够 {qa.get('out_of_time', 0)}）"
    )
    lines.append(f"  降级：{'、'.join(d['degraded']) or '无'}")
    lat = d["latency_ms"]
    lines.append(
        "  耗时："
        + " · ".join(f"{STAGE_LABELS[k]} {_ms(lat.get(k))}" for k in STAGE_LABELS if k in lat)
    )
    llm = d["llm"]
    if llm is not None:
        rows = "、".join(
            f"{row['label']} {row['calls']} 次"
            + (f"（失败 {row['failed']}）" if row["failed"] else "")
            for row in llm["by_purpose"]
        )
        lines.append(
            f"  大模型：{rows or '没有调用'} · token 输入 {llm['input_tokens']} / 输出 "
            f"{llm['output_tokens']}"
        )
    for c in d["checks"]:
        kind = "必查" if c["must"] else "参考"
        detail = f"：{c['detail']}" if c["detail"] else ""
        lines.append(f"  {STATUS_MARKS[c['status']]} {kind} {c['label']}{detail}")
    return "\n".join(lines)


def render_summary(report: dict[str, Any]) -> str:
    s = report["summary"]
    chain = report["chain"]
    lines = [f"== 汇总（{s['scenarios']} 个场景）· {chain['badge']}"]
    if chain.get("relaxed_note"):
        lines.append(f"  ! {chain['relaxed_note']}")
    for c in report["platform_checks"]:
        detail = f"：{c['detail']}" if c["detail"] else ""
        lines.append(f"  {STATUS_MARKS[c['status']]} 必查 {c['label']}{detail}")
    lines.append("  耗时（P50 / P95，对照 §12.1）：")
    for st in s["latency"].values():
        target = f"目标 ≤ {_ms(st['target_ms'])}" if st["target_ms"] else "没有单列目标"
        mark = {True: "✓", False: "✗", None: " "}[st["within_target"]]
        lines.append(f"    {mark} {st['label']}：{_ms(st['p50'])} / {_ms(st['p95'])}（{target}）")
    deg = s["degradations"]
    if deg["route_fallback"] and chain.get("judge") == "llm" and chain.get("latency") == "strict":
        lines.append(
            f"  ! {deg['route_fallback']} 条消息的路由退回了关键词规则"
            "（这些消息的判断层没用上大模型）；失败原因以超时为主时，演示可以加 --latency relaxed"
        )
    lines.append(
        f"  降级：路由退回关键词规则 {deg['route_fallback']} 次；"
        f"质检不可用 {deg['qa_unavailable']} 次；质检时间不够 {deg['qa_out_of_time']} 次"
        + (
            "；" + "、".join(f"{k} {n} 次" for k, n in deg["labels"].items())
            if deg["labels"]
            else ""
        )
    )
    llm = s["llm"]
    if llm["calls"]:
        calls = "、".join(f"{PURPOSE_LABELS.get(k, k)} {n} 次" for k, n in llm["calls"].items())
        failed = "、".join(
            f"{PURPOSE_LABELS.get(k, k)} {n} 次" for k, n in llm["failed"].items() if n
        )
        lines.append(
            f"  大模型调用：{calls}"
            + (f"（失败 {failed}）" if failed else "")
            + f" · token 输入 {llm['input_tokens']} / 输出 {llm['output_tokens']}"
        )
    m = s["must"]
    lines.append(
        f"  必查：{m['passed']}/{m['total']} 通过，{m['failed']} 项没通过，"
        f"{m['skipped']} 项不适用；参考：{s['info']['passed']}/{s['info']['total']}"
    )
    if s["errors"]:
        lines.append(f"  ✗ {s['errors']} 个场景出错")
    failed_checks = [
        f"[{PLATFORM_NAMES[Platform(r['platform'])]}] {r['scenario']}：{c['label']}"
        for r in report["scenarios"]
        for c in r["checks"]
        if c["must"] and c["status"] == "fail"
    ]
    for line in failed_checks:
        lines.append(f"    ✗ {line}")
    lines.append("自检通过" if s["ok"] else "自检没通过：有必查项失败")
    return "\n".join(lines)


__all__ = [
    "Check",
    "ScenarioResult",
    "render_scenario",
    "render_summary",
    "run_scenario",
    "run_selfcheck",
    "summarize",
]

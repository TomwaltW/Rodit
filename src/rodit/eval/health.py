"""评测报告里的运行健康度（health）：这一轮评测算不算"评到了模型"（方案 §13.2、§12.2）。

方案 §13.2 的上线门槛（测试集事实正确、合规满分率都 ≥ 95%）只在"模型真的被评到"时才有意义。
接真实大模型跑评测时，调用失败、被截断、读不出，链路会按 §12.2 降级：路由退回关键词规则、
质检只跑硬规则、生成拿不到可用草稿。评测照样跑完、照样给出一组数字，但那组数字量的是兜底逻辑，
不是模型。这里定下两件事：报告里 health 这一块长什么样（RunHealth），和判定"这一轮算不算数"的
唯一一条规则（assess）。

- 谁填：rodit.eval.cli，只有 --llm env 才填；离线替身不调大模型，EvalCliReport.health 是 None。
- 谁执行：rodit.eval.fingerprint.verdict_from_report（python -m rodit.ops record-eval）用
  DEFAULT_CEILING 重新算一遍 assess，不信报告里的 valid。

纯数据和纯函数：只用 pydantic 和标准库，不 import rodit 的其他模块。这一块里只有计数、比例和
失败原因的标签（error_label），没有买家数据、没有密钥、没有模型服务的错误消息原文。
json_object_count 是评测和连通性检查共用的小工具：一条回答里顶层的 JSON 对象不止一个时算"用不上"。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CEILING = 0.02
"""出问题的用例占比上限：超过就不算"评到了模型"。record-eval 固定用这个值。"""
LOST_VOTE_CEILING = 0.10
"""丢票（route_lost_votes）的上限。这些用例的路由仍然出自判断层，所以比上面那个宽；
但票数少了，置信度的刻度就不一样（§13.3），太多了这一轮同样不算数。"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PurposeHealth(_Model):
    """一个用途（LLMRequest.purpose：route / qa / generate）的调用情况。"""

    calls: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)  # 调用抛了 LLMError / LLMTimeout
    timeouts: int = Field(default=0, ge=0)  # 其中超时
    # HTTP 200，但 finish_reason 是 length（被 max_tokens 截断）
    truncated: int = Field(default=0, ge=0)
    empty: int = Field(default=0, ge=0)  # HTTP 200，但文本为空
    # HTTP 200，但这个环节的解析器读不出东西（判断层：这一票作废；生成：解析不出回答）
    unusable: int = Field(default=0, ge=0)
    # 失败原因 → 次数，键由 error_label 生成
    errors: dict[str, int] = Field(default_factory=dict)


class HealthCounts(_Model):
    """一组用例（全部，或只看测试集）的计数，单位都是用例数。"""

    cases: int = Field(ge=0)
    # 路由没用上配置的判断层（降级标记里有 decision）
    route_fallback: int = Field(default=0, ge=0)
    # 路由用了判断层，但至少有一票没算上（调用失败、被截断、读不出）
    route_lost_votes: int = Field(default=0, ge=0)
    # 有草稿送过质检判断层的用例数（qa_unavailable 的分母）
    qa_judged: int = Field(default=0, ge=0)
    # 质检判断层不可用：送检的草稿一条也没拿到判断层的结论（降级标记里有 decision:qa，而且这条
    # 用例的质检调用没有一次读得出来）。两份草稿里只有一份没判成的不算：另一份是判断层判过的
    qa_unavailable: int = Field(default=0, ge=0)
    # 调过大模型生成的用例数（generation_unusable 的分母）
    generation_attempted: int = Field(default=0, ge=0)
    # 调了生成，却没有大模型草稿送到质检：生成的调用失败（第一种话术失败时生成器不用后面的），
    # 或输出全都解析不了。被时间预算截断的不在这里，见 budget_unjudged
    generation_unusable: int = Field(default=0, ge=0)
    # 引擎异常：链路上有一环出了意外（降级标记里有 engine、router、understanding、retriever、
    # generator、qa、storage、knowledge、audit、tenant_isolation 之一，
    # 见 eval.cli.PIPELINE_ERROR_CODES）
    engine_errors: int = Field(default=0, ge=0)
    # 被时间预算截断（降级标记里有 budget）：照实展示，本身不算"模型没被评到"
    budget_cut: int = Field(default=0, ge=0)
    # 其中没有一条大模型草稿拿到质检判断层结论的（截在生成或质检完成之前）：这一条量到的不是
    # 模型的回答，超过上限这一轮不算数。已经有草稿判过、只是重写被截断的不在这里
    budget_unjudged: int = Field(default=0, ge=0)


class RunHealth(_Model):
    """一轮评测的运行健康度（EvalCliReport.health）。"""

    ceiling: float = Field(ge=0, le=1)  # 这次判定用的上限（比例）
    overall: HealthCounts
    # 测试集（§13.2 门槛看的那部分）；这次没有测试集用例时为 None
    test: HealthCounts | None = None
    by_purpose: dict[str, PurposeHealth] = Field(default_factory=dict)
    # 实际做出路由的判断层 → 用例数（incloud_llm / rules / classifier / none）
    routed_by: dict[str, int] = Field(default_factory=dict)
    # handle() 返回之后才结束的调用数（已经计入 by_purpose）
    late_calls: int = Field(default=0, ge=0)
    valid: bool  # 这一轮算不算"评到了模型"：assess 按上面的 ceiling 没有给出原因
    # valid 为 False 时的原因，给人看的中文句子
    reasons: list[str] = Field(default_factory=list)


def allowed(n: int, ceiling: float) -> int:
    """n 条里最多容许几条出问题：n ≤ 0 时是 0，否则 max(1, floor(ceiling × n))。

    至少容许 1 条：偶发的一次限流或超时不该让跑了几十分钟的一轮作废。"""
    if n <= 0:
        return 0
    return max(1, math.floor(ceiling * n))


def _share(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total > 0 else "—"


def _reasons(c: HealthCounts, ceiling: float, lost_vote_ceiling: float) -> list[str]:
    out: list[str] = []

    def check(count: int, total: int, cap: float, stage: str, among: str, what: str) -> None:
        limit = allowed(total, cap)
        if count > limit:
            share = _share(count, total)
            out.append(f"{stage}：{among}有 {count} 条（{share}）{what}，上限 {limit} 条。")

    cases = f"{c.cases} 条里"
    judged = f"有草稿送检的 {c.qa_judged} 条里"
    attempted = f"调过大模型生成的 {c.generation_attempted} 条里"
    fallback = "没用上判断层（退回关键词规则）"
    no_judge = "质检判断层不可用（只跑了硬规则）"
    no_draft = "没有大模型草稿送到质检（生成调用失败，或输出解析不了）"
    cut = "被时间预算截断、没有大模型草稿过完质检"
    lost = "至少有一票没算上（调用失败、被截断、读不出）"
    check(c.route_fallback, c.cases, ceiling, "路由", cases, fallback)
    check(c.qa_unavailable, c.qa_judged, ceiling, "质检", judged, no_judge)
    check(c.generation_unusable, c.generation_attempted, ceiling, "生成", attempted, no_draft)
    check(c.engine_errors, c.cases, ceiling, "引擎", cases, "引擎异常")
    check(c.budget_unjudged, c.cases, ceiling, "时间预算", cases, cut)
    check(c.route_lost_votes, c.cases, lost_vote_ceiling, "路由", cases, lost)
    return out


def assess(
    overall: HealthCounts,
    test: HealthCounts | None = None,
    *,
    ceiling: float = DEFAULT_CEILING,
    lost_vote_ceiling: float = LOST_VOTE_CEILING,
) -> list[str]:
    """这一轮算不算"评到了模型"：返回不算数的原因，空列表表示算数。

    全部用例看一遍，给了 test 时测试集再单独看一遍（句子前面加"测试集："）。超过上限就有一条原因：
    - route_fallback、engine_errors、budget_unjudged：上限 allowed(cases, ceiling)；
    - qa_unavailable：上限 allowed(qa_judged, ceiling)；
    - generation_unusable：上限 allowed(generation_attempted, ceiling)；
    - route_lost_votes：上限 allowed(cases, lost_vote_ceiling)。
    budget_cut 本身不产生原因（§12.1 的时间预算是正常行为，照实展示就行）；其中连一条大模型草稿都
    没判完的（budget_unjudged）才算：那几条量到的不是模型的回答。每句都写明条数、分母、占比和上限，
    例如"路由：324 条里有 105 条（32.4%）没用上判断层（退回关键词规则），上限 6 条。"
    """
    reasons = _reasons(overall, ceiling, lost_vote_ceiling)
    if test is not None:
        reasons += [f"测试集：{r}" for r in _reasons(test, ceiling, lost_vote_ceiling)]
    return reasons


def json_object_count(text: str) -> int:
    """一段模型输出里有几个顶层的 JSON 对象（能解析、不是空对象的才算；嵌在别的对象里的、
    字符串里的花括号不算，解析不了的花括号跳过）。

    要求输出 JSON 的请求，正常的回答里只有一个。有两个以上（比如先写了一份草稿、后面才是最终
    答案）时解析器取的未必是模型最后给的那个：评测把这样的回答算作"用不上"，连通性检查也不放行。"""
    decoder, count, i = json.JSONDecoder(), 0, 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return count
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(value, dict) and value:
            count += 1
        i = end


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def error_label(detail: Mapping[str, Any]) -> str:
    """一次失败调用的 llm.usage 审计 detail → PurposeHealth.errors 的键。

    按顺序：kind 是 timeout → "timeout"；有 status 而且不是 2xx → "http_<状态码>"，有 code 时再接
    ":<code>"；有别的 kind → kind（status 是 2xx 时有 code 也接上：HTTP 200 但响应读不出来、或
    响应体里带着错误对象，标成 "malformed" / "malformed:<code>"，不标成 "http_200"）；都没有 →
    detail["error"]（异常类型名），连它也没有 → "unknown"。"""
    kind = _text(detail.get("kind"))
    if kind == "timeout":
        return "timeout"
    status = detail.get("status")
    code = _text(detail.get("code"))
    if isinstance(status, int) and not isinstance(status, bool):
        if not 200 <= status < 300:
            return f"http_{status}:{code}" if code else f"http_{status}"
        if kind:
            return f"{kind}:{code}" if code else kind
        return f"http_{status}:{code}" if code else f"http_{status}"
    return kind or _text(detail.get("error")) or "unknown"


__all__ = [
    "DEFAULT_CEILING",
    "LOST_VOTE_CEILING",
    "HealthCounts",
    "PurposeHealth",
    "RunHealth",
    "allowed",
    "assess",
    "error_label",
    "json_object_count",
]

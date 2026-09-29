"""评测用例的数据结构和构造工具（方案 §13.1），dataset.py 和各 cases_*.py 共用。

标注约定（每条用例）：
- expected_intent / acceptable_intents：正确的问题类型，和同样说得通的备选类型（§5.1）；
- must_cite：每个元素是"任选其一"的依据 ID 组，推荐回答的依据里至少要有组内一个；
- must_include：回答里必须出现的要点（子串）；
- must_not_include：回答里不能出现的内容（错误事实、违规话术、买家的个人信息）；
- expected_actions：允许的路由结果（空表示按问题类型正常回答，即 answer / high_risk_answer /
  split_multi）；
- no_suggestion_ok：不给推荐也算对（投诉、要求人工、依据不足时，§5.7）；
- holdout：是否留作测试集（§13.1 留出 20%，不参与调参）。None 表示按 id 哈希分组——只有初版
  81 条用例是这样；扩充的用例都显式标注（见 dataset.py 的分组说明）。

tags 常用取值：high_risk、multi（一条消息多个问题）、reference（依赖历史的追问）、pii（文字里有
买家个人信息）、emotion、typo（错别字 / 谐音）、colloquial（口语）、knowledge_gap（依据里没有答案，
不该编）、no_item_card / no_order_card（没带卡片）、ambiguous_order、conflict（低优先级知识和政策
冲突）、forbidden_claim、human、confusion，以及对应核心流程的 §11.1–§11.5。
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rodit.core.enums import Intent, Role, RouteAction

Split = Literal["dev", "test"]


def hash_split(case_id: str) -> Split:
    """初版用例的分组规则：id 的 sha256 模 5 余 0 进测试集（约 20%）。"""
    h = int(hashlib.sha256(case_id.encode()).hexdigest(), 16)
    return "test" if h % 5 == 0 else "dev"


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    expected_intent: Intent
    acceptable_intents: list[Intent] = Field(default_factory=list)
    shop: str = "SHOP_A"  # fixtures 里的常量名
    buyer: str = "BUYER"  # fixtures 里的常量名
    item_id: str | None = None
    sku_id: str | None = None
    order_id: str | None = None
    history: list[tuple[Role, str]] = Field(default_factory=list)
    expected_actions: list[RouteAction] = Field(default_factory=list)
    must_cite: list[list[str]] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    no_suggestion_ok: bool = False
    tags: list[str] = Field(default_factory=list)
    holdout: bool | None = None

    @property
    def split(self) -> Split:
        """测试集（§13.1 留出，不参与调参）或开发集。"""
        if self.holdout is None:
            return hash_split(self.id)
        return "test" if self.holdout else "dev"


# ---------------------------------------------------------------- 写用例用的简写

I, A, R = Intent, RouteAction, Role  # noqa: E741

HR = [A.HIGH_RISK_ANSWER]  # 高风险类型：推荐 + 风险提示（§6.6 规则 4）
HR_SOOTHE = [A.HIGH_RISK_ANSWER, A.SOOTHE]  # 高风险售后，买家情绪可能达到安抚线（规则 3）
HR_OR_ANSWER = [A.HIGH_RISK_ANSWER, A.ANSWER]  # 备选类型不是高风险时，正常回答也对
HUMAN = [A.HUMAN_PRIORITY]  # 规则 2 / 投诉：提醒客服优先处理，不给推荐（§5.1、§5.7）
HUMAN_SOOTHE = [A.HUMAN_PRIORITY, A.SOOTHE]
SOOTHE_OR_HUMAN = [A.SOOTHE, A.HUMAN_PRIORITY]  # 情绪激烈（规则 3），也可能先命中要求人工（规则 2）
SOOTHE_OR_ANSWER = [A.SOOTHE, A.ANSWER]  # 有情绪但不一定到安抚线
OFF = [A.OFF_PLATFORM_SCRIPT]  # 规则 1：平台规则标准话术（§11.4）
MULTI = [A.SPLIT_MULTI, A.ANSWER]  # 规则 5：按问题分条推荐；一条回答把几个问题都答全也算对
ASK = [A.ANSWER, A.CLARIFY]  # 缺商品 / 订单 / 信息时先澄清（§5.2.4）
OPEN = [A.ANSWER, A.CLARIFY, A.NO_SUGGESTION]  # other 类"视情况"（§5.1）

OFF_RULE = "platform:taobao:off_platform"  # 站外交易标准话术的依据（§5.1 平台规则模板）


def case(id: str, text: str, intent: Intent, *, test: bool = False, **kw) -> EvalCase:  # noqa: A002
    """扩充用例：显式标注测试集归属（test=True 进测试集）。"""
    return EvalCase(id=id, text=text, expected_intent=intent, holdout=test, **kw)


__all__ = [
    "A",
    "ASK",
    "HR",
    "HR_OR_ANSWER",
    "HR_SOOTHE",
    "HUMAN",
    "HUMAN_SOOTHE",
    "MULTI",
    "OFF",
    "OFF_RULE",
    "OPEN",
    "SOOTHE_OR_ANSWER",
    "SOOTHE_OR_HUMAN",
    "EvalCase",
    "I",
    "R",
    "Split",
    "case",
    "hash_split",
]

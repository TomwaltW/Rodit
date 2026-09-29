"""扩充评测集：其他（方案 §5.1 greeting / off_platform / other，§5.7 转人工，§11.4 站外交易）。

- 问候 / 催回复用模板；问客服时间引用店铺政策"客服在线时间 9:00–23:00"；
- 站外交易请求一律推荐平台规则标准话术（§6.6 规则 1，优先级最高），依据是平台规则 · 站外交易，
  标准话术里说明"平台规定不能通过站外联系和交易"（§11.4）；回答里不能出现买家留的联系方式；
- 要求人工、连续两轮没听懂（§5.7）提醒客服优先处理，不给推荐；
- 其他问题"视情况"：答、澄清或不推荐都算对，但不能编。
"""

from __future__ import annotations

from rodit.eval.cases_base import HUMAN, OFF, OFF_RULE, OPEN, A, EvalCase, I, R, case

_OFF = {"expected_actions": OFF, "must_cite": [[OFF_RULE]], "must_include": ["站外"]}

OTHER_CASES: list[EvalCase] = [
    # ================================================ 问候 / 催回复 greeting
    case("greet-03", "亲 在不在", I.GREETING, tags=["colloquial"]),
    case("greet-04", "hello", I.GREETING),
    case(
        "greet-05",
        "怎么不回消息啊",
        I.GREETING,
        # 催回复（§5.1 greeting）；有点不耐烦，可以带安抚
        expected_actions=[A.ANSWER, A.SOOTHE],
    ),
    case(
        "greet-06",
        "客服几点下班",
        I.GREETING,
        acceptable_intents=[I.OTHER],
        must_cite=[["policy:service_hours"]],
        must_include=["23:00"],
    ),
    case(
        "greet-07",
        "晚上十点还有人吗",
        I.GREETING,
        test=True,
        acceptable_intents=[I.OTHER],
        must_cite=[["policy:service_hours"]],
        must_include=["23:00"],
    ),
    case(
        "greet-08",
        "好的 谢谢亲",
        I.GREETING,
        acceptable_intents=[I.OTHER],
        expected_actions=[A.ANSWER, A.NO_SUGGESTION],
        no_suggestion_ok=True,
    ),
    case(
        "greet-09",
        "？？？",
        I.GREETING,
        test=True,
        acceptable_intents=[I.OTHER],
        expected_actions=OPEN,
        no_suggestion_ok=True,
    ),
    # ================================================ 站外交易 off_platform（规则 1）
    case("off-05", "加我微信聊吧", I.OFF_PLATFORM, tags=["§11.4"], **_OFF),
    case("off-06", "vx多少", I.OFF_PLATFORM, tags=["typo"], **_OFF),
    case("off-07", "威信号发我一下", I.OFF_PLATFORM, tags=["typo"], **_OFF),
    case("off-08", "我们线下见面交易吧", I.OFF_PLATFORM, **_OFF),
    case("off-09", "发个支付宝账号 我直接转你", I.OFF_PLATFORM, tags=["§11.4"], **_OFF),
    case("off-10", "银行卡转账可以吗 便宜点就行", I.OFF_PLATFORM, **_OFF),
    case(
        "off-11",
        "我微信是 wx_demo_2026 加我",
        I.OFF_PLATFORM,
        must_not_include=["wx_demo_2026"],
        tags=["pii"],
        **_OFF,
    ),
    case("off-12", "给个电话呗 有事好联系", I.OFF_PLATFORM, tags=["colloquial"], **_OFF),
    case(
        "off-13",
        "加个微信吧，另外这裙子有M码吗",
        I.OFF_PLATFORM,
        item_id="i_555",
        acceptable_intents=[I.STOCK],
        tags=["multi", "§11.4"],
        **_OFF,
    ),
    case(
        "off-14",
        "拍下先不付款，私下转你行不",
        I.OFF_PLATFORM,
        test=True,
        tags=["colloquial"],
        **_OFF,
    ),
    # ================================================ 其他 other
    case(
        "other-02",
        "你们是机器人吗",
        I.OTHER,
        expected_actions=OPEN,
        no_suggestion_ok=True,
    ),
    case(
        "other-03",
        "有实体店吗",
        I.OTHER,
        expected_actions=OPEN,
        no_suggestion_ok=True,
        tags=["knowledge_gap"],
    ),
    case(
        "other-04",
        "招代理吗 可以批发吗",
        I.OTHER,
        test=True,
        expected_actions=OPEN,
        no_suggestion_ok=True,
        tags=["knowledge_gap"],
    ),
    case(
        "other-05",
        "今天天气真好",
        I.OTHER,
        acceptable_intents=[I.GREETING],
        expected_actions=OPEN,
        no_suggestion_ok=True,
    ),
    case(
        "other-06",
        "嗯嗯",
        I.OTHER,
        test=True,
        acceptable_intents=[I.GREETING],
        expected_actions=OPEN,
        no_suggestion_ok=True,
    ),
    case(
        "other-07",
        "人工客服",
        I.OTHER,
        acceptable_intents=[I.COMPLAINT, I.GREETING],
        expected_actions=HUMAN,
        no_suggestion_ok=True,
        tags=["human"],
    ),
    case(
        "other-08",
        "找个真人跟我说",
        I.OTHER,
        acceptable_intents=[I.COMPLAINT, I.GREETING],
        expected_actions=HUMAN,
        no_suggestion_ok=True,
        tags=["human"],
    ),
    case(
        "other-09",
        "还是没听懂，你到底在说啥",
        I.OTHER,
        test=True,
        order_id="o_1002",
        history=[
            (R.BUYER, "我的快递到哪了"),
            (R.AGENT, "亲，截至 10:30 已到达杭州转运中心哦"),
            (R.BUYER, "啥意思 没懂"),
            (R.AGENT, "亲，就是包裹现在在杭州中转呢"),
        ],
        acceptable_intents=[I.LOGISTICS_TRACK, I.COMPLAINT],
        # 买家连续两轮表示没听懂：提示客服接管（§5.7）
        expected_actions=HUMAN,
        no_suggestion_ok=True,
        tags=["confusion", "reference", "human"],
    ),
    case(
        "other-10",
        "裙子太好看了爱了爱了",
        I.OTHER,
        item_id="i_555",
        acceptable_intents=[I.GREETING, I.PRODUCT_SPEC],
        # 正面情绪不该触发安抚
        expected_actions=OPEN,
        no_suggestion_ok=True,
        tags=["emotion"],
    ),
    case(
        "other-11",
        "asdfghjkl",
        I.OTHER,
        acceptable_intents=[I.GREETING],
        expected_actions=OPEN,
        no_suggestion_ok=True,
    ),
    case(
        "ctx-15",
        "好的那我拍了",
        I.OTHER,
        item_id="i_555",
        history=[(R.BUYER, "黑色M有货吗"), (R.AGENT, "亲，黑色M有货的哦")],
        acceptable_intents=[I.GREETING],
        expected_actions=[A.ANSWER, A.NO_SUGGESTION],
        no_suggestion_ok=True,
        tags=["reference"],
    ),
]

"""三个平台的原始消息 → 标准消息（方案 §8.3）：buyer_ref 哈希、昵称等个人信息丢弃、
商品 / 订单卡片、去重、非买家事件、时间戳、平台接入模式限制（§8.1）。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rodit.adapters import (
    DouyinMockAdapter,
    NullToolGateway,
    PddMockAdapter,
    TaobaoMockAdapter,
    cst_from_epoch_ms,
    cst_from_epoch_s,
)
from rodit.core.clock import CST
from rodit.core.enums import Mode, Platform, Role
from rodit.core.models import ItemRef, OrderRef
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import (
    AGENT,
    BUYER,
    BUYER_DOUYIN,
    BUYER_PDD,
    CONV,
    DEFAULT_NOW,
    RAW_BUYERS,
    SHOP_A,
    SHOP_B,
    SHOP_C,
    SHOP_D,
)


@pytest.fixture
def taobao(world, clock, hasher) -> TaobaoMockAdapter:
    return TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B")


@pytest.fixture
def douyin(world, clock, hasher) -> DouyinMockAdapter:
    return DouyinMockAdapter(InMemoryToolGateway(world, Platform.DOUYIN), clock, hasher, access="B")


@pytest.fixture
def pdd_adapter(clock, hasher) -> PddMockAdapter:
    return PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access="C")


def _assert_no_pii(msg, pii, raw_buyer_id: str) -> None:
    dumped = msg.model_dump_json()
    for value in (pii["nick"], pii["phone"], "文三路", raw_buyer_id):
        assert value not in dumped


# ================================================================ 淘宝


def test_taobao_text_message_to_standard(taobao, tb):
    msg = taobao.on_message(tb())
    assert msg is not None
    assert msg.platform == Platform.TAOBAO
    assert (msg.shop_id, msg.conversation_id, msg.msg_id) == (SHOP_A, CONV, "tb_m_0001")
    assert msg.buyer_ref == BUYER  # 原始买家 ID 用同一个盐哈希，和演示订单对得上
    assert msg.text == "165 110斤穿什么码"
    assert msg.item_card == ItemRef(item_id="i_555")
    assert msg.order_card is None
    assert msg.agent_id == AGENT  # "店铺名:子账号" 取子账号
    assert msg.role == Role.BUYER and msg.mode == Mode.ASSIST
    assert msg.received_at == DEFAULT_NOW and msg.received_at.utcoffset() == timedelta(hours=8)
    assert msg.extension == {"msg_type": "text", "reception": "human"}


def test_taobao_drops_nickname_phone_address(taobao, tb, pii):
    raw = tb(
        receiver_name="张三",
        receiver_mobile=pii["phone"],
        receiver_address=pii["address"],
        buyer_mobile=pii["phone"],
    )
    msg = taobao.on_message(raw)
    assert msg is not None
    _assert_no_pii(msg, pii, RAW_BUYERS["BUYER"][1])
    assert "张三" not in msg.model_dump_json()
    assert set(msg.extension) <= {"msg_type", "reception"}


def test_taobao_item_and_order_cards(taobao, tb):
    item = taobao.on_message(tb(msg_id="m_item", msg_type="item", content="", sku_id="k_1"))
    assert item is not None and item.text == ""
    assert item.item_card == ItemRef(item_id="i_555", sku_id="k_1")
    order = taobao.on_message(
        tb(msg_id="m_order", msg_type="order", content="", item_id=None, tid="o_1002")
    )
    assert order is not None and order.order_card == OrderRef(order_id="o_1002")
    assert order.item_card is None
    # 数字 ID 也接受（平台常用长整型）
    num = taobao.on_message(tb(msg_id="m_num", item_id=555, tid=1002))
    assert num.item_card.item_id == "555" and num.order_card.order_id == "1002"


def test_taobao_dedup_per_shop(taobao, tb):
    assert taobao.on_message(tb()) is not None
    assert taobao.on_message(tb()) is None  # 平台重推
    assert taobao.on_message(tb(content="改了内容也算重复")) is None
    # 同一个 msg_id 在另一个店铺是另一条消息（按 (shop_id, msg_id) 去重）
    other = taobao.on_message(tb(seller_id=SHOP_B))
    assert other is not None and other.shop_id == SHOP_B


def test_dedup_lru_is_bounded(world, clock, hasher, tb):
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="B", dedup_capacity=2
    )
    for mid in ("m1", "m2", "m3"):
        assert adapter.on_message(tb(msg_id=mid)) is not None
    assert adapter.on_message(tb(msg_id="m3")) is None
    assert adapter.on_message(tb(msg_id="m1")) is not None  # 已被淘汰，容量有界


def test_taobao_non_buyer_events_return_none(taobao, tb):
    assert taobao.on_message(tb(msg_id="b1", content="在吗")) is not None
    echo = tb(msg_id="a1", direction="out", content="亲，在的~", buyer_open_uid=None)
    assert taobao.on_message(echo) is None
    for mt in ("system", "image", "voice", "video", "unknown"):
        assert taobao.on_message(tb(msg_id=f"x_{mt}", msg_type=mt)) is None
    assert taobao.on_message(tb(msg_id="d1", direction="sideways")) is None
    # 客服回声记进本地上下文（role=agent），不进引擎
    ctx = taobao.get_context(SHOP_A, CONV)
    assert [(m.msg_id, m.role) for m in ctx.recent_messages] == [
        ("b1", Role.BUYER),
        ("a1", Role.AGENT),
    ]


def test_taobao_recall_removes_message_from_context(taobao, tb):
    taobao.on_message(tb(msg_id="b1", content="第一句"))
    taobao.on_message(tb(msg_id="b2", content="发错了"))
    assert taobao.on_message(tb(msg_id="r1", msg_type="recall", recall_msg_id="b2")) is None
    assert [m.msg_id for m in taobao.get_context(SHOP_A, CONV).recent_messages] == ["b1"]


@pytest.mark.parametrize(
    "over",
    [
        {"buyer_open_uid": None},  # 没有买家 ID 生成不了 buyer_ref
        {"buyer_open_uid": ""},
        {"msg_id": ""},
        {"seller_id": None},
        {"session_id": "   "},
        {"send_time_ms": "not-a-number"},
        {"send_time_ms": -5},
        {"send_time_ms": True},
        {"content": "", "item_id": None},  # 没有文字也没有卡片
        {"content": 12345, "item_id": None},
    ],
)
def test_taobao_unrecognizable_payloads(taobao, tb, over):
    assert taobao.on_message(tb(**over)) is None


def test_not_a_mapping_returns_none(taobao):
    assert taobao.on_message(None) is None  # type: ignore[arg-type]
    assert taobao.on_message(["not", "a", "dict"]) is None  # type: ignore[arg-type]


def test_invalid_payload_does_not_poison_dedup(taobao, tb):
    """解析失败的消息不占用 msg_id：平台修正后重推同一个 msg_id 仍能处理。"""
    assert taobao.on_message(tb(buyer_open_uid=None)) is None
    assert taobao.on_message(tb()) is not None


def test_missing_timestamp_uses_clock(taobao, tb, clock):
    raw = tb()
    del raw["send_time_ms"]
    clock.advance(minutes=3)
    assert taobao.on_message(raw).received_at == clock.now()


def test_taobao_mode_auto_only_for_access_a_robot_reception(world, clock, hasher, tb):
    auto = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="A", auto_send_enabled=True
    )
    assert auto.on_message(tb(msg_id="r", reception="robot")).mode == Mode.AUTO
    assert auto.on_message(tb(msg_id="h", reception="human")).mode == Mode.ASSIST
    assist_a = TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="A")
    assert assist_a.on_message(tb(msg_id="r", reception="robot")).mode == Mode.ASSIST
    plugin = TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B")
    assert plugin.on_message(tb(msg_id="r", reception="robot")).mode == Mode.ASSIST


def test_other_demo_buyers_hash_to_demo_refs(taobao, tb):
    from rodit.testing.fixtures import BUYER_R3, BUYER_RISKY

    risky = taobao.on_message(tb(msg_id="x1", buyer_open_uid=RAW_BUYERS["BUYER_RISKY"][1]))
    r3 = taobao.on_message(tb(msg_id="x2", buyer_open_uid=RAW_BUYERS["BUYER_R3"][1]))
    assert (risky.buyer_ref, r3.buyer_ref) == (BUYER_RISKY, BUYER_R3)


def test_taobao_buyer_ref_matches_demo_orders(world, taobao, tb):
    """buyer_ref 能直接用来查该买家在本店的最近订单（工具层用同一个 hasher）。"""
    msg = taobao.on_message(tb())
    orders = taobao.tools().list_recent_orders(msg.shop_id, msg.buyer_ref)
    assert orders and all(o.buyer_ref == BUYER for o in orders)


# ================================================================ 抖店（飞鸽插件）


def test_douyin_text_message_uses_plugin_context(douyin, dy, pii):
    msg = douyin.on_message(dy())
    assert msg is not None
    assert msg.platform == Platform.DOUYIN
    assert (msg.shop_id, msg.conversation_id, msg.msg_id) == (
        SHOP_D,
        "dyc_001",
        "7290000000000001",
    )
    assert msg.buyer_ref == BUYER_DOUYIN
    assert msg.item_card == ItemRef(item_id="i_d1")  # 来自插件看到的当前商品
    assert msg.agent_id == "kf_d01"
    assert msg.received_at == DEFAULT_NOW
    assert msg.extension == {"content_type": "text", "source": "feige_plugin"}
    _assert_no_pii(msg, pii, RAW_BUYERS["BUYER_DOUYIN"][1])


def test_douyin_cards_in_content_override_context(douyin, dy):
    product = douyin.on_message(
        dy(
            server_message_id="p1",
            content={"type": "product", "product_id": "i_d9", "sku_id": "k_9"},
        )
    )
    assert product.item_card == ItemRef(item_id="i_d9", sku_id="k_9") and product.text == ""
    order = douyin.on_message(
        dy(server_message_id="o1", content={"type": "order", "order_id": "o_d001"})
    )
    assert order.order_card == OrderRef(order_id="o_d001")
    assert order.item_card == ItemRef(item_id="i_d1")


def test_douyin_non_buyer_events(douyin, dy):
    assert douyin.on_message(dy(server_message_id="b1")) is not None
    cs = dy(
        server_message_id="cs1",
        sender={"role": "customer_service", "uid": "kf_d01"},
        content={"type": "text", "text": "亲，48 小时内发货"},
    )
    assert douyin.on_message(cs) is None
    system = dy(server_message_id="s1", sender={"role": "system"})
    assert douyin.on_message(system) is None
    image = dy(server_message_id="i1", content={"type": "image", "url": "x"})
    assert douyin.on_message(image) is None
    withdraw = dy(server_message_id="w1", content={"type": "withdraw", "withdraw_msg_id": "b1"})
    assert douyin.on_message(withdraw) is None
    ctx = douyin.get_context(SHOP_D, "dyc_001")
    assert [(m.msg_id, m.role) for m in ctx.recent_messages] == [("cs1", Role.AGENT)]


@pytest.mark.parametrize(
    "over",
    [
        {"sender": "customer"},  # 嵌套字段类型不对
        {"content": ["text"]},
        {"sender": {"role": "customer"}},  # 没有 uid
        {"create_time": 10**12},  # 用了毫秒：超出范围
    ],
)
def test_douyin_unrecognizable_payloads(douyin, dy, over):
    assert douyin.on_message(dy(**over)) is None


@pytest.mark.parametrize("access", ["A", "C"])
def test_douyin_only_supports_plugin_mode(world, clock, hasher, access):
    tools = (
        InMemoryToolGateway(world, Platform.DOUYIN)
        if access != "C"
        else NullToolGateway(Platform.DOUYIN)
    )
    with pytest.raises(ValueError, match="只支持接入模式"):
        DouyinMockAdapter(tools, clock, hasher, access=access)


# ================================================================ 拼多多


def test_pdd_text_message(pdd_adapter, pdd, pii):
    msg = pdd_adapter.on_message(pdd())
    assert msg is not None
    assert msg.platform == Platform.PDD
    assert (msg.shop_id, msg.conversation_id, msg.msg_id) == (SHOP_C, "pddc_001", "pdd_m_0001")
    assert msg.buyer_ref == BUYER_PDD
    assert msg.item_card == ItemRef(item_id="i_c1")
    assert msg.agent_id == "kf_c01"
    assert msg.received_at == DEFAULT_NOW
    assert msg.extension == {"msg_type_code": 1}
    _assert_no_pii(msg, pii, RAW_BUYERS["BUYER_PDD"][1])


def test_pdd_type_codes(pdd_adapter, pdd):
    goods = pdd_adapter.on_message(pdd(msg_id="g", type="11", content="", goods_id="i_c2"))
    assert goods.item_card == ItemRef(item_id="i_c2") and goods.text == ""
    order = pdd_adapter.on_message(
        pdd(msg_id="o", type=12, content="", goods_id=None, order_sn="260928-001")
    )
    assert order.order_card == OrderRef(order_id="260928-001")
    assert pdd_adapter.on_message(pdd(msg_id="img", type=2)) is None
    assert pdd_adapter.on_message(pdd(msg_id="sys", type=31)) is None
    assert pdd_adapter.on_message(pdd(msg_id="bad", type="abc")) is None
    assert pdd_adapter.on_message(pdd(msg_id="bool", type=True)) is None
    cs = pdd(msg_id="cs", **{"from": {"role": "mall_cs", "uid": "kf_c01"}})
    assert pdd_adapter.on_message(cs) is None
    assert pdd_adapter.on_message(pdd(msg_id="rc", type=1002, recall_msg_id="g")) is None
    assert [m.msg_id for m in pdd_adapter.get_context(SHOP_C, "pddc_001").recent_messages] == [
        "o",
        "cs",
    ]


def test_pdd_is_mode_c_only_without_platform_data(world, clock, hasher):
    with pytest.raises(ValueError, match="NullToolGateway"):
        PddMockAdapter(InMemoryToolGateway(world, Platform.PDD), clock, hasher, access="C")
    for access in ("A", "B"):
        with pytest.raises(ValueError, match="只支持接入模式"):
            PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access=access)
    adapter = PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access="C")
    tools = adapter.tools()
    assert isinstance(tools, NullToolGateway)
    assert tools.get_order(SHOP_C, "o_1001") is None


# ================================================================ 时间戳


def test_epoch_helpers_return_cst():
    assert cst_from_epoch_ms(int(DEFAULT_NOW.timestamp() * 1000)) == DEFAULT_NOW
    assert cst_from_epoch_s(str(int(DEFAULT_NOW.timestamp()))) == DEFAULT_NOW
    assert cst_from_epoch_s(DEFAULT_NOW.timestamp()).tzinfo == CST
    for bad in (None, True, "x", 0, -1, float("nan"), float("inf"), 10**12):
        with pytest.raises(ValueError):
            cst_from_epoch_s(bad)

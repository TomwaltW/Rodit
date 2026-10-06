"""接入层：买家图片消息（方案 §15 Phase 3 图片理解、§8.3 标准会话格式、§5.3 最小化个人信息）。

淘宝（msg_type=image + pic_key）和抖店（content.type=image + content.images[].uri）的图片消息
→ StandardMessage.images；只收不透明的平台图片标识，带签名的网址一律不往后传。
"""

from __future__ import annotations

from typing import Any

import pytest

from rodit.adapters import DouyinMockAdapter, TaobaoMockAdapter
from rodit.adapters.base import MAX_IMAGES_PER_MESSAGE, caption, image_ref
from rodit.core.enums import Mode, Platform, Role
from rodit.core.models import ImageRef, ItemRef, OrderRef, StandardMessage
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import CONV, SHOP_A, SHOP_B, SHOP_D

SIGNED_URL = "https://img.alicdn.example.invalid/bao/x.jpg?Expires=1&Signature=tok_secret_123"


@pytest.fixture
def taobao(world, clock, hasher) -> TaobaoMockAdapter:
    return TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B")


@pytest.fixture
def douyin(world, clock, hasher) -> DouyinMockAdapter:
    return DouyinMockAdapter(InMemoryToolGateway(world, Platform.DOUYIN), clock, hasher, access="B")


def tb_image(tb: Any, pic: Any = "img_buyer_damage_1", **over: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "msg_type": "image",
        "content": "[图片]",
        "pic_key": pic,
        "pic_url": SIGNED_URL,
        "item_id": None,
    }
    fields.update(over)
    return tb(**fields)


def dy_image(dy: Any, *uris: Any, text: str = "", **over: Any) -> dict[str, Any]:
    images = [{"uri": u, "url": SIGNED_URL} for u in uris]
    return dy(content={"type": "image", "text": text, "images": images}, **over)


# ================================================================ 图片标识检查


@pytest.mark.parametrize(
    "value",
    [
        "img_buyer_damage_1",
        "tfs/TB1abc.jpg",
        "bucket:2026/09/28/abc-123.png",
        "7290000000000001",
        "photo1.jpg",  # 后缀不是域名
    ],
)
def test_image_ref_accepts_opaque_keys(value: str):
    assert image_ref(value) == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        123.5,
        True,
        ["img_1"],
        SIGNED_URL,
        "http://img.example.invalid/a.jpg",
        "//img.example.invalid/a.jpg",
        "www.example.com/a.jpg",
        "img.alicdn.com/bao/a.jpg",  # 没写协议的网址
        "cdn.example.cn:8080/a.jpg",
        "img_1.jpg?token=abc",
        "img 1.jpg",
        "_leading_underscore",
        "img_13812345678",  # 标识里夹着手机号：不往后传
        "a" * 201,
        "图片1",
    ],
)
def test_image_ref_rejects_urls_and_junk(value: Any):
    assert image_ref(value) is None


def test_image_ref_accepts_numeric_ids():
    assert image_ref(7290000000000001) == "7290000000000001"
    assert image_ref(" img_1 ") == "img_1"
    # 平台分配的长数字 ID 不当银行卡号挡掉（这个号码能过 Luhn 校验）
    assert image_ref("4111111111111111") == "4111111111111111"
    # 像手机号的纯数字不要
    assert image_ref("13812345678") is None
    assert image_ref(13812345678) is None


@pytest.mark.parametrize(("value", "expected"), [("[图片]", ""), ("【图片】", ""), ("[IMAGE]", "")])
def test_caption_drops_platform_placeholders(value: str, expected: str):
    assert caption(value) == expected
    assert caption("  收到就是这样的  ") == "收到就是这样的"
    assert caption(None) == ""


# ================================================================ 淘宝


def test_taobao_image_only_message(taobao, tb):
    msg = taobao.on_message(tb_image(tb))
    assert isinstance(msg, StandardMessage)
    assert msg.text == ""
    assert msg.images == [ImageRef(ref="img_buyer_damage_1")]
    assert msg.extension == {"msg_type": "image", "reception": "human"}
    assert msg.item_card is None and msg.order_card is None
    # 带签名的临时地址不进标准消息的任何字段
    dumped = msg.model_dump_json()
    assert "Signature" not in dumped and "alicdn" not in dumped and "tok_secret" not in dumped


def test_taobao_image_with_caption_and_cards(taobao, tb):
    raw = tb_image(tb, content="收到就是这样的", item_id="i_555", tid="o_1013")
    msg = taobao.on_message(raw)
    assert msg.text == "收到就是这样的"
    assert msg.images == [ImageRef(ref="img_buyer_damage_1")]
    assert msg.item_card == ItemRef(item_id="i_555") and msg.order_card == OrderRef(
        order_id="o_1013"
    )


@pytest.mark.parametrize("pic", [None, "", SIGNED_URL, "img.alicdn.com/a.jpg", "img 1"])
def test_taobao_image_without_usable_key_is_ignored(taobao, tb, pic):
    raw = tb_image(tb, pic=pic, content="其实我是图片消息")
    assert taobao.peek_shop_id(raw) is None
    assert taobao.on_message(raw) is None
    # 没有进本地上下文
    assert taobao.get_context(SHOP_A, CONV).recent_messages == []


def test_taobao_image_message_without_pic_key_field(taobao, tb):
    raw = tb(msg_type="image")  # 旧格式：没有 pic_key，照旧忽略
    assert taobao.on_message(raw) is None


def test_taobao_image_local_context_and_dedup(taobao, tb):
    assert taobao.on_message(tb_image(tb, msg_id="p1")) is not None
    assert taobao.on_message(tb_image(tb, msg_id="p1")) is None  # 平台重推：按 msg_id 去重
    ctx = taobao.get_context(SHOP_A, CONV)
    assert [(m.msg_id, m.role, m.text) for m in ctx.recent_messages] == [("p1", Role.BUYER, "")]


def test_taobao_agent_image_echo_is_not_recorded(taobao, tb):
    taobao.on_message(tb(msg_id="b1", content="在吗"))
    echo = tb_image(tb, msg_id="a1", direction="out", buyer_open_uid=None)
    assert taobao.on_event(echo) is None
    ctx = taobao.get_context(SHOP_A, CONV)
    assert [m.msg_id for m in ctx.recent_messages] == ["b1"]


def test_taobao_image_from_robot_reception_in_mode_a(world, clock, hasher, tb):
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="A", auto_send_enabled=True
    )
    msg = adapter.on_message(tb_image(tb, reception="robot"))
    assert msg.mode == Mode.AUTO and msg.images == [ImageRef(ref="img_buyer_damage_1")]


def test_taobao_image_events_stay_in_their_shop(taobao, tb):
    a = taobao.on_message(tb_image(tb, msg_id="p1"))
    b = taobao.on_message(tb_image(tb, msg_id="p1", seller_id=SHOP_B))  # 同一 msg_id、另一家店
    assert a.shop_id == SHOP_A and b.shop_id == SHOP_B
    assert taobao.peek_shop_id(tb_image(tb, msg_id="p2", seller_id=SHOP_B)) == SHOP_B


def test_taobao_image_recall_removes_it_from_context(taobao, tb):
    taobao.on_message(tb_image(tb, msg_id="p1"))
    recall = tb(msg_id="r1", msg_type="recall", recall_msg_id="p1")
    assert taobao.on_event(recall) is not None
    assert taobao.get_context(SHOP_A, CONV).recent_messages == []


# ================================================================ 抖店


def test_douyin_image_message_multiple_images(douyin, dy):
    msg = douyin.on_message(dy_image(dy, "img_buyer_damage_1", "img_buyer_wrong_1"))
    assert msg.shop_id == SHOP_D and msg.text == ""
    assert [i.ref for i in msg.images] == ["img_buyer_damage_1", "img_buyer_wrong_1"]
    assert msg.extension == {"content_type": "image", "source": "feige_plugin"}
    assert msg.item_card == ItemRef(item_id="i_d1")  # 插件看到的当前商品照常带上
    assert "Signature" not in msg.model_dump_json()


def test_douyin_image_caption_dedupe_and_bad_entries(douyin, dy):
    raw = dy(
        content={
            "type": "image",
            "text": "这是收到的样子",
            "images": [
                {"uri": "img_buyer_damage_1"},
                {"uri": "img_buyer_damage_1"},  # 重复
                {"url": SIGNED_URL},  # 只有网址
                {"uri": SIGNED_URL},  # 标识位置上是网址
                {"uri": "img_buyer_size_1"},
            ],
        }
    )
    msg = douyin.on_message(raw)
    assert msg.text == "这是收到的样子"
    assert [i.ref for i in msg.images] == ["img_buyer_damage_1", "img_buyer_size_1"]


def test_douyin_image_caps_at_nine(douyin, dy):
    uris = [f"img_{i:02d}" for i in range(12)]
    msg = douyin.on_message(dy_image(dy, *uris))
    assert len(msg.images) == MAX_IMAGES_PER_MESSAGE == 9
    assert [i.ref for i in msg.images] == uris[:9]


@pytest.mark.parametrize(
    "content",
    [
        {"type": "image", "url": "x"},  # 旧格式：只有网址
        {"type": "image", "images": []},
        {"type": "image", "images": [{"url": SIGNED_URL}]},
        {"type": "image", "images": "img_1"},  # 格式不对
        {"type": "image", "images": ["img_1"]},  # 列表里不是对象
    ],
)
def test_douyin_image_without_usable_key_is_ignored(douyin, dy, content):
    raw = dy(content=content)
    assert douyin.on_message(raw) is None
    assert douyin.peek_shop_id(raw) is None


def test_douyin_customer_service_image_is_not_recorded(douyin, dy):
    raw = dy_image(
        dy,
        "img_cs_1",
        server_message_id="cs1",
        sender={"role": "customer_service", "uid": "kf_d01"},
    )
    assert douyin.on_event(raw) is None
    assert douyin.get_context(SHOP_D, "dyc_001").recent_messages == []

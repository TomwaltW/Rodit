"""接入层测试的公共夹具：演示世界、假时钟、测试盐的 hasher、三个平台的合成原始消息。"""

from __future__ import annotations

from typing import Any

import pytest

from rodit.core.pii import BuyerRefHasher
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import (
    CONV,
    DEFAULT_NOW,
    RAW_BUYERS,
    SHOP_A,
    SHOP_C,
    SHOP_D,
    TEST_SALT,
    DemoWorld,
    build_demo_world,
)

NOW_MS = int(DEFAULT_NOW.timestamp() * 1000)
NOW_S = int(DEFAULT_NOW.timestamp())

# 合成的个人信息：只用来断言它们不会出现在标准消息里
NICK = "小美爱穿搭"
PHONE = "13812345678"
ADDRESS = "浙江省杭州市西湖区文三路 100 号 3 幢 502 室"


@pytest.fixture
def world() -> DemoWorld:
    return build_demo_world()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def hasher() -> BuyerRefHasher:
    return BuyerRefHasher(TEST_SALT)


def taobao_raw(**over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": "tb_m_0001",
        "msg_type": "text",
        "direction": "in",
        "buyer_open_uid": RAW_BUYERS["BUYER"][1],
        "buyer_nick": NICK,
        "content": "165 110斤穿什么码",
        "item_id": "i_555",
        "sku_id": None,
        "sub_account": "小雅女装旗舰店:kf_02",
        "send_time_ms": NOW_MS,
        "reception": "human",
    }
    raw.update(over)
    return raw


def douyin_raw(**over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "shop_id": SHOP_D,
        "conversation_short_id": "dyc_001",
        "server_message_id": "7290000000000001",
        "sender": {"role": "customer", "uid": RAW_BUYERS["BUYER_DOUYIN"][1], "nickname": NICK},
        "content": {"type": "text", "text": "这条裙子什么时候发货"},
        "context": {"product_id": "i_d1", "sku_id": None, "order_id": None},
        "create_time": NOW_S,
        "service_uid": "kf_d01",
        "source": "feige_plugin",
    }
    raw.update(over)
    return raw


def pdd_raw(**over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "mall_id": SHOP_C,
        "conv_id": "pddc_001",
        "msg_id": "pdd_m_0001",
        "type": 1,
        "from": {"role": "buyer", "uid": RAW_BUYERS["BUYER_PDD"][1]},
        "nickname": NICK,
        "content": "打底衫是纯棉的吗",
        "goods_id": "i_c1",
        "ts": NOW_S,
        "cs_account": "kf_c01",
    }
    raw.update(over)
    return raw


@pytest.fixture
def tb() -> Any:
    """淘宝原始消息构造函数：tb(msg_id="x", content="…")。"""
    return taobao_raw


@pytest.fixture
def dy() -> Any:
    """抖店（飞鸽插件）原始消息构造函数。"""
    return douyin_raw


@pytest.fixture
def pdd() -> Any:
    """拼多多（多多客服）原始消息构造函数。"""
    return pdd_raw


@pytest.fixture
def pii() -> dict[str, str]:
    """合成的个人信息样本（只用来断言它们不会出现在标准消息里）。"""
    return {"nick": NICK, "phone": PHONE, "address": ADDRESS}

"""演示站的买家图片（方案 §15 Phase 3 "图片理解：破损照片、尺码截图"、§7.3.1、§5.3、§8.1）：
三个图片场景走真实的问答引擎（图片理解用合成数据的替身），聊天接口只收固定的合成图片标识，
拼多多模式 C 不能发图片；物流截图上的手机号在任何响应里都不出现（合成号码）。"""

from __future__ import annotations

import json

import pytest
from showcase_support import Showcase, build_showcase

from rodit.core.enums import Platform
from rodit.engine.images import ALERT_VISION_FAILED
from rodit.showcase.app import API_PREFIX, page_config
from rodit.showcase.images import (
    DEMO_IMAGE_REFS,
    IMAGE_PLATFORMS,
    MAX_CHAT_IMAGES,
    NO_IMAGE_REASON,
    UNREADABLE_REF,
)
from rodit.showcase.scenarios import scenarios_for
from rodit.testing.fixtures import DEMO_IMAGE_FINDINGS

API = API_PREFIX
PHONE = "13812345678"
"""合成物流截图（img_buyer_logistics_1）面单上的手机号：哪里都不能出现原文。"""

IMAGE_SCENARIOS = ("photo_damage", "photo_size", "photo_logistics")
ON = sorted(IMAGE_PLATFORMS)


def _cards(view: dict) -> list[dict]:
    return view["result"]["suggestions"]


def _post_chat(showcase: Showcase, **body):
    return showcase.post(f"{API}/chat", body)


# ================================================================ 三个图片场景


@pytest.mark.parametrize("platform", ON)
def test_damage_photo_gets_the_photo_template(showcase: Showcase, platform):
    """§7.3.1 收到的东西坏了：文字 + 破损照片 → 质量问题（高风险，规则 4），第一条是确定性模板
    （照片看到了、请在订单页申请售后并上传照片），图片理解只是参考。"""
    view = showcase.chat(platform, "photo_damage")
    bubble = view["message"]
    assert bubble["text"] == "收到货发现破了，能退吗？"
    assert [i["ref"] for i in bubble["images"]] == ["img_buyer_damage_1"]
    assert bubble["images"][0]["title"] == "破损照片"
    u = view["trace"]["understand"]
    assert u["intent"] == "quality_issue" and u["rule"] == 4 and u["high_risk"]
    first = _cards(view)[0]
    assert first["kind"] == "template"
    assert "您发来的照片我们看到了" in first["text"] and "申请售后" in first["text"]
    assert "上传" in first["text"] and "退款" not in first["text"]  # 不承诺处理结果
    assert "请买家在申请售后时上传照片凭证" in first["risk_hints"]
    assert not any(c["prefill"] for c in _cards(view))  # 高风险类型不半自动
    images = view["trace"]["images"]
    assert images["total"] == 1 and images["failed"] == 0 and images["degraded"] is False
    [finding] = images["findings"]
    assert finding["ref"] == "img_buyer_damage_1"
    assert finding["kind"] == "damage" and finding["kind_label"] == "破损"
    assert finding["confidence"] == 0.93 and finding["routed"] is True
    assert finding["summary"] == DEMO_IMAGE_FINDINGS["img_buyer_damage_1"]["summary"]
    assert images["description"].startswith("【买家发来图片：破损（外包装一角压扁")
    [alert] = images["alerts"]
    assert alert.startswith("买家发来 1 张图片：破损（93%）") and alert.endswith("以凭证为准")
    assert alert in view["result"]["alerts"]


def test_damage_photo_is_on_the_order_of_aftersale_as_2009(showcase: Showcase):
    """同一笔订单（o_1013，同一位合成买家）后来申请了售后：售后审核助手里的 as_2009 带 3 张凭证。"""
    scenarios = showcase.get(f"{API}/scenarios", platform="taobao").json()["scenarios"]
    damage = next(s for s in scenarios if s["id"] == "photo_damage")
    assert damage["card"]["kind"] == "order" and damage["card"]["id"] == "o_1013"
    assert damage["card"]["status"] == "已签收"
    assert [i["ref"] for i in damage["images"]] == ["img_buyer_damage_1"]
    assert "as_2009" in damage["expect"]
    item = showcase.aftersale("taobao", "as_2009")
    assert item["order_id"] == "o_1013"


@pytest.mark.parametrize("platform", ON)
def test_size_screenshot_alone_gets_size_advice(showcase: Showcase, platform):
    """§15 Phase 3 尺码截图：买家只发图不打字，图中文字交给路由和生成，按尺码建议推荐 M。"""
    view = showcase.chat(platform, "photo_size")
    assert view["message"]["text"] == "" and len(view["message"]["images"]) == 1
    u = view["trace"]["understand"]
    assert u["intent"] == "size_advice" and u["rule"] == 6
    [card] = _cards(view)
    assert "M" in card["text"] and "160–165" in card["text"]
    assert [e["label"] for e in card["evidence"]] == ["商品属性 · 尺码建议"]
    [finding] = view["trace"]["images"]["findings"]
    assert finding["kind_label"] == "尺码截图" and finding["confidence"] == 0.95
    assert finding["extracted_text"] == "身高 165cm 体重 110 斤"
    assert "图中文字：身高 165cm 体重 110 斤" in view["trace"]["images"]["description"]


@pytest.mark.parametrize("platform", ON)
def test_logistics_screenshot_is_masked_and_answered_from_tracking(showcase: Showcase, platform):
    """§5.3：物流截图面单上的手机号在识别结果里就已打码；物流截图的图中文字不交给路由和大模型，
    回答按实时物流轨迹写明"截至"。"""
    view = showcase.chat(platform, "photo_logistics")
    [finding] = view["trace"]["images"]["findings"]
    assert finding["kind_label"] == "物流 / 订单截图"
    assert finding["extracted_text"] == "中通快递 已签收 联系电话 [手机号]"
    description = view["trace"]["images"]["description"]
    assert "联系电话" not in description and "[手机号]" not in description
    u = view["trace"]["understand"]
    assert u["intent"] == "logistics_track" and u["rule"] == 6
    [card] = _cards(view)
    assert card["kind"] == "template" and "截至" in card["text"] and "驿站" in card["text"]
    assert [e["id"] for e in card["evidence"]] == ["tool:logistics:o_1003"]
    assert PHONE not in json.dumps(view, ensure_ascii=False)


def test_raw_phone_number_never_appears_in_any_showcase_response():
    """合成物流截图上的手机号：所有接口响应、页面本身、单元里存下的推荐和审计都没有原文。"""
    sc = build_showcase()
    sc.get("/")
    for platform in Platform:
        p = str(platform)
        sc.get(f"{API}/scenarios", platform=p)
        for s in scenarios_for(platform):
            sc.chat(p, s.id)
        if platform in IMAGE_PLATFORMS:
            sc.chat(p, "photo_logistics", "看看这个", ["img_buyer_logistics_1"])
            sc.chat(p, "photo_size", None, ["img_buyer_logistics_1", "img_buyer_size_1"])
        sc.get(f"{API}/scenarios", platform=p)
        sc.aftersales(p)
        sc.dashboard(p)
        sc.get(f"{API}/overview", platform=p)
    assert PHONE not in sc.all_text()
    assert "[手机号]" in sc.all_text()  # 打码后的识别结果确实展示了
    for platform in IMAGE_PLATFORMS:
        demo = sc.hub.get(platform)
        stored = [r.model_dump_json() for r in demo.unit.repos.suggestions.list(demo.shop_id)]
        stored += [e.model_dump_json() for e in demo.unit.repos.audit.list_events(demo.shop_id)]
        assert stored and not [s for s in stored if PHONE in s]


# ================================================================ 降级、多张图、只发文字


@pytest.mark.parametrize("platform", ON)
def test_unreadable_image_degrades_and_still_answers_the_text(showcase: Showcase, platform):
    """§12.2：图片识别失败 → 记降级、提醒客服自己看图，照常按文字出推荐。"""
    view = showcase.chat(platform, "photo_size", "这个什么时候发货", [UNREADABLE_REF])
    images = view["trace"]["images"]
    assert images["total"] == 1 and images["failed"] == 1 and images["findings"] == []
    assert images["degraded"] is True and ALERT_VISION_FAILED in images["alerts"]
    assert "图片理解（没能识别，请客服自己看图）" in view["result"]["degraded"]
    assert view["trace"]["understand"]["intent"] == "shipping_policy"
    [card] = _cards(view)
    assert any("图片没能识别" in h for h in card["risk_hints"])
    assert card["prefill"] is False  # 有降级、有提醒：不半自动
    only = showcase.chat(platform, "photo_damage", None, [UNREADABLE_REF])
    assert _cards(only) == [] and only["result"]["no_suggestion_reason"]


@pytest.mark.parametrize("platform", ON)
def test_several_images_are_read_in_order_as_one_turn(showcase: Showcase, platform):
    """淘宝每张图一条消息、抖店一条消息带几张图：都合并成一轮处理，识别结果按买家发的顺序。"""
    refs = ["img_buyer_damage_1", "img_buyer_size_1"]
    view = showcase.chat(platform, "photo_size", "这两张你看下", refs)
    assert [i["ref"] for i in view["message"]["images"]] == refs
    images = view["trace"]["images"]
    assert images["total"] == 2
    assert [f["ref"] for f in images["findings"]] == refs
    assert images["description"].startswith("【买家发来 2 张图片：破损")
    demo = showcase.hub.get(platform)
    assert demo.unit.vision.calls[-2:] == [(demo.shop_id, r) for r in refs]


def test_typed_text_in_an_image_scenario_sends_text_only(showcase: Showcase):
    view = showcase.chat("taobao", "photo_size", "那 170 120斤呢")
    assert view["message"]["images"] == [] and view["trace"]["images"] is None
    assert "L" in _cards(view)[0]["text"]


# ================================================================ 接口校验、模式 C


@pytest.mark.parametrize(
    "images",
    [
        ["img_made_up"],
        ["https://img.example.invalid/a.jpg"],
        ["img_buyer_damage_1", "img_buyer_damage_1"],
        ["img_buyer_damage_1", "img_buyer_size_1", "img_buyer_wrong_1", "img_buyer_product_1"],
        [],
        "img_buyer_damage_1",
        [1],
    ],
)
def test_chat_images_must_come_from_the_synthetic_whitelist(showcase: Showcase, images):
    r = _post_chat(showcase, platform="taobao", scenario="photo_size", images=images)
    assert r.status_code == 422 and r.json()["error"] == "invalid", r.text
    assert showcase.hub.built() == []  # 校验在接口层：没有组装单元，更没有调图片理解


def test_whitelist_is_the_synthetic_findings_plus_unreadable():
    assert DEMO_IMAGE_REFS == set(DEMO_IMAGE_FINDINGS) | {"img_unreadable"}
    assert MAX_CHAT_IMAGES == 3


@pytest.mark.parametrize("ref", sorted(DEMO_IMAGE_REFS))
def test_every_whitelisted_image_is_accepted_on_taobao(showcase: Showcase, ref):
    r = _post_chat(showcase, platform="taobao", scenario="photo_damage", images=[ref])
    assert r.status_code == 200, r.text
    assert r.json()["trace"]["images"]["total"] == 1


def test_pdd_mode_c_cannot_send_images(showcase: Showcase):
    """§8.1：拼多多是模式 C 话术助手，客服只能粘贴文字，图片转发不进来。"""
    r = _post_chat(showcase, platform="pdd", scenario="size", images=["img_buyer_size_1"])
    assert r.status_code == 422
    assert r.json() == {"error": "invalid", "message": NO_IMAGE_REASON}
    pdd = showcase.get(f"{API}/scenarios", platform="pdd").json()
    assert pdd["can_send_images"] is False and pdd["no_image_reason"] == NO_IMAGE_REASON
    assert all(s["images"] == [] for s in pdd["scenarios"])
    # 图片场景在拼多多不存在
    assert _post_chat(showcase, platform="pdd", scenario="photo_size").status_code == 404
    assert showcase.hub.get(Platform.PDD).unit.vision.calls == []


def test_scenarios_and_page_describe_the_image_tiles(showcase: Showcase):
    taobao = showcase.get(f"{API}/scenarios", platform="taobao").json()
    assert taobao["can_send_images"] is True and taobao["no_image_reason"] is None
    by_id = {s["id"]: s for s in taobao["scenarios"]}
    assert by_id["photo_size"]["text"] == "" and by_id["photo_size"]["images"][0]["icon"] == "size"
    config = page_config(showcase.hub)
    tiles = {t["ref"]: t for t in config["images"]}
    assert set(tiles) == DEMO_IMAGE_REFS
    assert {r for r, t in tiles.items() if t["picker"]} == {
        "img_buyer_damage_1",
        "img_buyer_size_1",
        "img_buyer_wrong_1",
        "img_buyer_logistics_1",
        "img_buyer_product_1",
        "img_unreadable",
    }
    assert all(t["title"] and t["caption"] and t["icon"] for t in tiles.values())
    assert config["max_chat_images"] == 3 and config["no_image_reason"] == NO_IMAGE_REASON
    assert [p["can_send_images"] for p in config["platforms"]] == [True, True, False]


# ================================================================ 售后凭证（§7.3.2 R5）


@pytest.mark.parametrize("platform", list(Platform))
def test_r5_card_shows_the_evidence_image_summary(showcase: Showcase, platform):
    """as_2009（质量问题、3 张凭证）：R5 的建议理由后面接图片理解的摘要（只是参考，建议仍是
    人工决定），页面上单独显示，另给凭证图片的张数和占位图块。"""
    item = showcase.aftersale(platform, "as_2009")
    review = item["review"]
    assert [r["id"] for r in review["rules"]] == ["R5"]
    assert review["suggestion"] == "manual"
    assert review["reason_main"] == "质量 / 破损 / 错漏发，买家已上传 3 张凭证：客服查看凭证后决定"
    assert review["vision_summary"] == (
        "3 张凭证中 2 张显示破损（裙摆撕裂，线头外露；外包装压痕明显）"
    )
    assert (
        review["reason"]
        == review["reason_main"] + "；图片理解（仅供参考）：" + (review["vision_summary"])
    )
    assert item["evidence_count"] == 3
    assert [i["ref"] for i in item["evidence_images"]] == [
        "img_as2009_1",
        "img_as2009_2",
        "img_as2009_3",
    ]
    others = [i for i in showcase.aftersales(platform)["items"] if i["aftersale_id"] != "as_2009"]
    assert all(i["review"]["vision_summary"] is None for i in others)
    assert all(i["review"]["reason_main"] == i["review"]["reason"] for i in others)


def test_aftersale_page_points_to_the_production_desk_without_linking(showcase: Showcase):
    html = showcase.get("/").text
    assert "/aftersale" in html and "客服令牌登录" in html
    assert 'href="/aftersale' not in html  # 演示站里没有这个页面：只写文字，不放链接

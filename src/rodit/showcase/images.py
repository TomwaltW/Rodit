"""演示站的买家图片（方案 §15 Phase 3 "图片理解：破损照片、尺码截图"、§5.3、§8.1）。

- 只认一组固定的合成图片标识（DEMO_IMAGE_REFS）：rodit.testing.fixtures.DEMO_IMAGE_FINDINGS 的键，
  外加识别失败的 img_unreadable（不在表里，图片理解的替身按识别失败处理）。别的标识一律 422；
- 页面上不显示真实图片：演示站没有图片文件、不引用任何外部地址。每个标识对应一个带文字说明的
  占位图块（DEMO_IMAGE_TILES，页面用 CSS / SVG 画），说明的是"这张合成图片拍的是什么"；
  识别结果（类型、置信度、摘要、图中文字）以问答引擎里的图片理解（FakeImageUnderstanding，
  平台云内视觉模型的替身）给的为准，页面在"AI 是怎么想的"里展示；
- 买家只能在模式 B（淘宝千牛、抖店飞鸽的插件）的会话里发图片：插件能读到平台的图片消息。
  拼多多是模式 C 独立话术助手，客服只能把买家的文字粘贴进来（§8.1），不能转发图片。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rodit.core.enums import Platform
from rodit.testing.fixtures import DEMO_IMAGE_FINDINGS

UNREADABLE_REF = "img_unreadable"
"""识别失败的图（不在 DEMO_IMAGE_FINDINGS 里）：演示图片理解失败时的降级（§12.2）。"""

DEMO_IMAGE_REFS: frozenset[str] = frozenset(DEMO_IMAGE_FINDINGS) | {UNREADABLE_REF}
"""接口接受的图片标识（固定的合成标识，不收网址）。"""

MAX_CHAT_IMAGES = 3
"""一条买家消息最多带几张图（和问答引擎一次识别的上限 engine.images.MAX_IMAGES 一致）。"""

IMAGE_PLATFORMS: frozenset[Platform] = frozenset({Platform.TAOBAO, Platform.DOUYIN})
"""能发图片的平台（模式 B 插件读得到平台的图片消息；拼多多模式 C 只能粘贴文字）。"""

NO_IMAGE_REASON = (
    "拼多多是模式 C 独立话术助手：客服把买家的文字粘贴进来，图片转发不进来（§8.1），"
    "需要看图时由客服在多多客服里自己看"
)


@dataclass(frozen=True)
class ImageTile:
    """一张合成图片在页面上的占位图块。"""

    ref: str
    title: str  # 图块上的标题
    caption: str  # 这张合成图片拍的是什么（不是识别结果）
    icon: str  # 图块上画的图形：damage / size / wrong / logistics / product / blur / evidence
    picker: bool = True  # 是否出现在买家"发图片"的选择框里（售后凭证只在售后审核里出现）


DEMO_IMAGE_TILES: tuple[ImageTile, ...] = (
    ImageTile("img_buyer_damage_1", "破损照片", "包裹压扁、衣物撕裂的实拍", "damage"),
    ImageTile("img_buyer_size_1", "身高体重截图", "备忘录里写着身高、体重", "size"),
    ImageTile("img_buyer_wrong_1", "错发照片", "收到的颜色和下单的不一样", "wrong"),
    ImageTile(
        "img_buyer_logistics_1", "物流截图", "物流页面截图，面单上有收件人手机号", "logistics"
    ),
    ImageTile("img_buyer_product_1", "商品图", "从别处保存的一张连衣裙图片", "product"),
    ImageTile(UNREADABLE_REF, "模糊照片", "拍糊了、看不清（演示识别失败）", "blur"),
    ImageTile("img_as2009_1", "售后凭证 1", "买家申请售后时上传的凭证", "evidence", False),
    ImageTile("img_as2009_2", "售后凭证 2", "买家申请售后时上传的凭证", "evidence", False),
    ImageTile("img_as2009_3", "售后凭证 3", "买家申请售后时上传的凭证", "evidence", False),
)

TILES: dict[str, ImageTile] = {t.ref: t for t in DEMO_IMAGE_TILES}

if set(TILES) != DEMO_IMAGE_REFS:  # 合成数据加了图却没加占位图块：启动时就发现
    raise RuntimeError("演示站的图片占位图块和合成图片标识对不上")


def tile_view(ref: str) -> dict[str, Any]:
    """一张图在页面上的样子（标识、标题、说明、图形）。不认识的标识只给标识。"""
    tile = TILES.get(ref)
    if tile is None:
        return {"ref": ref, "title": "图片", "caption": "", "icon": "blur"}
    return {"ref": tile.ref, "title": tile.title, "caption": tile.caption, "icon": tile.icon}


def tiles_config() -> list[dict[str, Any]]:
    """写进页面配置的图块（服务端常量，没有店铺数据）。"""
    return [{**tile_view(t.ref), "picker": t.picker} for t in DEMO_IMAGE_TILES]


__all__ = [
    "DEMO_IMAGE_REFS",
    "DEMO_IMAGE_TILES",
    "IMAGE_PLATFORMS",
    "MAX_CHAT_IMAGES",
    "NO_IMAGE_REASON",
    "TILES",
    "UNREADABLE_REF",
    "ImageTile",
    "tile_view",
    "tiles_config",
]

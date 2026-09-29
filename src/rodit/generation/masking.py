"""进提示词 / 判断层请求前的个人信息最小化（方案 §5.3、§12.3）：core.pii.mask_pii 的薄包装。"""

from __future__ import annotations

from rodit.core.pii import mask_pii


def mask_text(text: str) -> str:
    """遮掉手机号、证件号、微信 / QQ 号、收件地址等；没有个人信息时原样返回。"""
    return mask_pii(text) if text else text

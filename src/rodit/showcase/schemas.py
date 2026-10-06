"""演示站接口的请求格式（pydantic 校验，不认识的字段一律拒绝；文字最多 500 字）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rodit.core.enums import AfterSaleAction, RejectReason
from rodit.showcase.images import DEMO_IMAGE_REFS, MAX_CHAT_IMAGES

MAX_TEXT_CHARS = 500
"""买家的话、编辑后的回答、打码演示的文字最多多少字。"""

MAX_BODY_BYTES = 16 * 1024
"""请求体最大字节数（超过返回 413）。"""

PLATFORM_PATTERN = r"^[a-z]{1,16}$"
ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ChatRequest(_Request):
    """text、images 都为空时说场景预设的话（带预设的图片）；给了任何一个就只发给的（只发图片、
    不带文字也可以）。images 只收固定的合成图片标识（showcase.images.DEMO_IMAGE_REFS），不重复。"""

    platform: str = Field(pattern=PLATFORM_PATTERN)
    scenario: str = Field(pattern=r"^[a-z_]{1,32}$")
    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    images: list[str] | None = Field(default=None, min_length=1, max_length=MAX_CHAT_IMAGES)

    @field_validator("images")
    @classmethod
    def _known_images(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = [ref for ref in v if ref not in DEMO_IMAGE_REFS]
        if unknown:
            raise ValueError("只能发演示用的合成图片")
        if len(set(v)) != len(v):
            raise ValueError("同一张图片不能重复发")
        return v


class FeedbackRequest(_Request):
    platform: str = Field(pattern=PLATFORM_PATTERN)
    suggestion_id: int = Field(ge=1, le=2**31 - 1)
    action: Literal["sent", "edited", "rejected"]
    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_CHARS)
    reason: RejectReason | None = None


class ConfirmRequest(_Request):
    platform: str = Field(pattern=PLATFORM_PATTERN)
    action: AfterSaleAction
    expected_status_version: str = Field(pattern=r"^[0-9A-Za-z._:-]{1,64}$")
    reason_code: str | None = Field(default=None, pattern=r"^[a-z_]{1,32}$")


class NegotiateRequest(_Request):
    platform: str = Field(pattern=PLATFORM_PATTERN)
    done: bool = False


class MaskRequest(_Request):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class ResetRequest(_Request):
    platform: str = Field(pattern=PLATFORM_PATTERN)


__all__ = [
    "ID_PATTERN",
    "MAX_CHAT_IMAGES",
    "MAX_BODY_BYTES",
    "MAX_TEXT_CHARS",
    "ChatRequest",
    "ConfirmRequest",
    "FeedbackRequest",
    "MaskRequest",
    "NegotiateRequest",
    "ResetRequest",
]

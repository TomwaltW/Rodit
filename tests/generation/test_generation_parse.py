"""解析大模型生成输出（方案 §5.4 JSON 格式；失败抛 LLMError，§12.2 退回模板）。"""

from __future__ import annotations

import pytest

from rodit.core.errors import LLMError
from rodit.generation import extract_json_object, parse_generation_output

GOOD = '{"answer": "亲，这款是棉 95%、氨纶 5% 的哦", "evidence_ids": ["item:i_555:attr:面料"]}'


def test_plain_json():
    out = parse_generation_output(GOOD)
    assert out.answer == "亲，这款是棉 95%、氨纶 5% 的哦"
    assert out.evidence_ids == ["item:i_555:attr:面料"]
    assert out.need_clarify is False


@pytest.mark.parametrize(
    "raw",
    [
        f"```json\n{GOOD}\n```",
        f"```\n{GOOD}\n```",
        f"好的，以下是回答：\n{GOOD}",
        f"{GOOD}\n以上回答只使用了依据。",
        f"按格式 {{answer}} 输出：```json\n{GOOD}\n``` 完毕",
    ],
)
def test_tolerates_fences_and_surrounding_text(raw):
    assert parse_generation_output(raw).answer.startswith("亲，这款是棉")


def test_braces_inside_strings():
    out = parse_generation_output('前言 {"answer": "规格 {M} 码", "evidence_ids": []} 后记')
    assert out.answer == "规格 {M} 码"


def test_prefers_object_with_answer():
    raw = '{"note": "草稿"} 最终：{"answer": "亲，在的", "evidence_ids": []}'
    assert parse_generation_output(raw).answer == "亲，在的"
    assert extract_json_object('{"x": 1}') == {"x": 1}


def test_need_clarify_and_lenient_fields():
    out = parse_generation_output(
        '{"answer": " 请问您平时穿什么码？ ", "evidence_ids": "item:i_555:size_chart",'
        ' "need_clarify": true, "confidence": 0.9}'
    )
    assert out.answer == "请问您平时穿什么码？"
    assert out.evidence_ids == ["item:i_555:size_chart"]
    assert out.need_clarify is True
    assert parse_generation_output('{"answer": "好的", "evidence_ids": null}').evidence_ids == []


def test_unknown_ids_are_kept_as_is():
    out = parse_generation_output('{"answer": "好的", "evidence_ids": ["faq:999", "faq:999"]}')
    assert out.evidence_ids == ["faq:999", "faq:999"]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "亲，明天一定能到哦！",
        "[1, 2, 3]",
        '{"evidence_ids": []}',
        '{"answer": ""}',
        '{"answer": "   "}',
        '{"answer": 123}',
        '{"answer": "好", "evidence_ids": [1, 2]}',
        '{"answer": "好", "need_clarify": "maybe"}',
        '{"answer": "未闭合',
    ],
)
def test_invalid_output_raises_llm_error(raw):
    with pytest.raises(LLMError):
        parse_generation_output(raw)


def test_error_message_does_not_echo_model_text():
    with pytest.raises(LLMError) as exc:
        parse_generation_output('{"answer": "", "note": "13812345678"}')
    assert "13812345678" not in str(exc.value)

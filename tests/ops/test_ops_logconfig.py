"""结构化 JSON 日志和过滤（方案 §12.3 日志脱敏、§5.3）：个人信息打码，令牌、连接串、密钥不出现。"""

from __future__ import annotations

import io
import json
import logging

import pytest

from rodit.ops.logconfig import (
    REDACTED,
    configure_logging,
    scrub,
    secrets_from_env,
)


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


@pytest.fixture
def stream() -> io.StringIO:
    out = io.StringIO()
    configure_logging("DEBUG", "json", secrets=["literal-secret-value-123"], stream=out)
    return out


@pytest.mark.parametrize(
    "text, leaked",
    [
        ("买家留了手机 13812345678 让回电", "13812345678"),
        ("寄到北京市朝阳区建国路88号3号楼2单元501", "建国路88号"),
        ("身份证 990101199003071237", "990101199003071237"),  # 合成：地区码 99 不存在
        ("邮箱 buyer@example.com", "buyer@example.com"),
        ("Authorization: Bearer abcdefghijklmnop123", "abcdefghijklmnop123"),
        ("令牌 rdt_AbCdEfGhIjKlMnOpQrStUv 失效", "rdt_AbCdEfGhIjKlMnOpQrStUv"),
        ("连接 postgresql://rodit_login:hunter2hunter2@db:5432/rodit 失败", "hunter2hunter2"),
        ("dsn host=db password=hunter2hunter2 dbname=rodit", "hunter2hunter2"),
        ("api_key=sk-abcdef0123456789", "sk-abcdef0123456789"),
        ("key is literal-secret-value-123!", "literal-secret-value-123"),
    ],
)
def test_scrub(text, leaked):
    cleaned = scrub(text, ["literal-secret-value-123"])
    assert leaked not in cleaned


def test_scrub_keeps_business_codes():
    text = "订单号 o_1002 运单号 SF1234567890123 店铺 s_1001 用时 1.5 秒"
    assert scrub(text) == text


def test_json_lines_with_whitelisted_extras(stream):
    log = logging.getLogger("rodit.test")
    log.info(
        "处理完成：店铺 %s 买家 13812345678",
        "s_1001",
        extra={"shop_id": "s_1001", "stage": "route", "body": {"text": "原文"}, "duration_ms": 12},
    )
    (line,) = _lines(stream)
    assert line["level"] == "INFO" and line["logger"] == "rodit.test"
    assert line["msg"].startswith("处理完成：店铺 s_1001 买家 ")
    assert "13812345678" not in line["msg"]
    assert line["shop_id"] == "s_1001" and line["stage"] == "route" and line["duration_ms"] == 12
    assert "body" not in line  # 白名单以外的 extra 不输出
    assert line["ts"].endswith("+00:00")


def test_exceptions_are_scrubbed(stream):
    log = logging.getLogger("rodit.test")
    try:
        raise ValueError("连接 postgresql://u:hunter2hunter2@db/rodit 失败，买家 13812345678")
    except ValueError:
        log.exception("出错了")
    (line,) = _lines(stream)
    assert line["exc_type"] == "ValueError"
    assert "Traceback" in line["exc"]
    assert "hunter2hunter2" not in line["exc"] and "13812345678" not in line["exc"]


def test_secret_env_values_are_redacted(monkeypatch):
    monkeypatch.setenv("RODIT_TOKEN_KEY", "synthetic-token-key-abcdef")
    monkeypatch.setenv("RODIT_BUYER_SALT", "abc")  # 太短：不当密钥替换
    out = io.StringIO()
    configure_logging("INFO", "json", stream=out)
    assert "synthetic-token-key-abcdef" in secrets_from_env()
    logging.getLogger("rodit.test").warning("key=%s", "synthetic-token-key-abcdef")
    logging.getLogger("rodit.test").warning("值 synthetic-token-key-abcdef 出现在句中")
    text = out.getvalue()
    assert "synthetic-token-key-abcdef" not in text
    assert REDACTED in text


def test_uvicorn_access_log_goes_through_filter(stream):
    access = logging.getLogger("uvicorn.access")
    access.addHandler(logging.NullHandler())  # uvicorn 自己的处理器会被清掉
    configure_logging("INFO", "json", stream=stream)
    access.info(
        '%s - "%s %s HTTP/%s" %d',
        "10.0.0.1:5000",
        "GET",
        "/v1/x?token=rdt_AbCdEfGhIjKlMnOpQrSt",
        "1.1",
        200,
    )
    (line,) = _lines(stream)
    assert line["logger"] == "uvicorn.access"
    assert "rdt_AbCdEfGhIjKlMnOpQrSt" not in line["msg"]
    assert not access.handlers and access.propagate


def test_configure_is_idempotent(stream):
    before = len(logging.getLogger().handlers)
    configure_logging("INFO", "json", stream=stream)
    configure_logging("INFO", "json", stream=stream)
    assert len(logging.getLogger().handlers) == before
    logging.getLogger("rodit.test").info("一次")
    assert len(_lines(stream)) == 1


def test_text_format_is_also_filtered():
    out = io.StringIO()
    configure_logging("INFO", "text", secrets=[], stream=out)
    logging.getLogger("rodit.test").info("电话 13812345678")
    assert "13812345678" not in out.getvalue() and "rodit.test" in out.getvalue()


def test_bad_format_rejected():
    with pytest.raises(ValueError):
        configure_logging("INFO", "xml", stream=io.StringIO())  # type: ignore[arg-type]


def test_bad_format_arguments_do_not_crash(stream):
    logging.getLogger("rodit.test").info("缺参数 %s %s", "只有一个")
    (line,) = _lines(stream)
    assert "缺参数" in line["msg"]


def test_uvicorn_levels_set_earlier_are_reset(stream):
    """uvicorn 以 log_level="warning"、access_log=False 启动过（同一进程里）：
    重新配置后访问日志照样经过过滤输出，不会被遗留的级别吞掉。"""
    access = logging.getLogger("uvicorn.access")
    access.setLevel(logging.WARNING)
    access.propagate = False
    configure_logging("INFO", "json", stream=stream)
    access.info('%s - "GET %s HTTP/1.1" 200', "10.0.0.1:5000", "/healthz")
    (line,) = _lines(stream)
    assert line["logger"] == "uvicorn.access" and "/healthz" in line["msg"]

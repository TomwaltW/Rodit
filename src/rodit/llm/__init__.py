"""大模型客户端实现（D5 境内已备案模型）。密钥只从环境变量读，仓库里没有任何密钥。

负责：T7。只依赖 rodit.core。
"""

from rodit.llm.metering import MeteredLLMClient
from rodit.llm.openai_compat import (
    LLMHTTPError,
    OpenAICompatibleClient,
    Transport,
    urllib_transport,
)

__all__ = [
    "LLMHTTPError",
    "MeteredLLMClient",
    "OpenAICompatibleClient",
    "Transport",
    "urllib_transport",
]

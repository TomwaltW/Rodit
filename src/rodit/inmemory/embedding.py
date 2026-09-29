"""离线可用的确定性向量：字符 n-gram 哈希（实现 core.ports.Embedder）。

真实向量模型接在同一个接口后面；这个实现用于测试、本地演示和没有向量服务时的兜底。
对短中文问句（FAQ）效果尚可：字面越接近分数越高，完全相同的句子分数为 1。
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence

_PUNCT = re.compile(r"[\s　-〿＀-／：-＠!-/:-@\[-`{-~]+")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text or "").lower())


def _bucket(token: str, dim: int) -> int:
    # 不能用内置 hash()：它每个进程随机化
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") % dim


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致：{len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """字符 1–3 gram 哈希到 dim 维，按 n 加权后 L2 归一化。"""

    def __init__(self, dim: int = 256, *, ngram_weights: tuple[float, ...] = (1.0, 1.5, 1.0)):
        if dim < 16:
            raise ValueError("dim 太小")
        self.dim = dim
        self._weights = ngram_weights

    def embed_one(self, text: str) -> list[float]:
        norm = _normalize(text)
        vec = [0.0] * self.dim
        for n, weight in enumerate(self._weights, start=1):
            for i in range(len(norm) - n + 1):
                vec[_bucket(f"{n}:{norm[i : i + n]}", self.dim)] += weight
        length = math.sqrt(sum(v * v for v in vec))
        return [v / length for v in vec] if length else vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

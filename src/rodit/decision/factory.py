"""按配置组装判断层实现（方案 §6.4、§6.6 decision.provider、D4）。

- cfg.provider 选路由用的实现：incloud_llm（MVP 默认）/ classifier（数据攒够后）/ rules / jev；
  classifier 用训练任务生成的模型文件（rodit.eval.train → ClassifierProvider.load，
  部署入口从 RODIT_CLASSIFIER_PATH 加载后通过 classifier 传进来）；测试和演示也可以传
  classifier_samples 现场训练；
- 质检（purpose="qa"）的问题只有大模型（或满足 §6.3 条件的 Jev）能答：
  provider 是 classifier / rules 时，质检仍用 incloud_llm（§6.4 "质检用硬规则 + 大模型"）；
- 投票次数：路由 cfg.vote_samples、质检 cfg.qa_vote_samples；
  多于一次时用 cfg.vote_temperature 采样，只采一次时温度取 0；
  每个问题要求过半数的有效票，否则按判断层不可用降级（§12.2）；
- 单次调用超时 cfg.timeout_s（路由 ≤ 1 秒，§12.1）。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from rodit.core.config import DecisionConfig
from rodit.core.decision import DecisionProvider, require_buyer_data_ok
from rodit.core.enums import Intent, ProviderName
from rodit.core.models import LabeledSample
from rodit.core.ports import LLMClient
from rodit.decision.classifier import ClassifierProvider
from rodit.decision.jev import JevProvider
from rodit.decision.llm import InCloudLLMProvider
from rodit.decision.rules import RuleProvider

JudgePurpose = Literal["route", "qa"]


def build_llm_provider(
    cfg: DecisionConfig, *, purpose: JudgePurpose, llm: LLMClient | None
) -> InCloudLLMProvider:
    if llm is None:
        raise ValueError("incloud_llm 判断层需要注入 LLMClient")
    samples = cfg.vote_samples if purpose == "route" else cfg.qa_vote_samples
    return InCloudLLMProvider(
        llm,
        samples=samples,
        temperature=cfg.vote_temperature if samples > 1 else 0.0,
        purpose=purpose,
        timeout_s=cfg.timeout_s,
        min_votes=samples // 2 + 1,
    )


def build_decision_provider(
    cfg: DecisionConfig,
    *,
    purpose: JudgePurpose,
    llm: LLMClient | None = None,
    classifier_samples: Iterable[LabeledSample] | Iterable[tuple[str, Intent]] | None = None,
    classifier: ClassifierProvider | None = None,
    jev: JevProvider | None = None,
) -> DecisionProvider:
    """按 cfg.provider 构造判断层。缺少所需依赖时抛 ValueError；Jev 不能处理买家数据时抛
    ComplianceError（配置校验已禁止未批准的 jev，§6.2）。

    provider=classifier 的路由：优先用传进来的 classifier（从模型文件加载好的），
    其次用 classifier_samples 现场训练（加 §5.1 例句）；两样都没有抛 ValueError。"""
    provider = cfg.provider
    if provider == ProviderName.JEV:
        if jev is None:
            raise ValueError("provider=jev 需要注入 JevProvider")
        require_buyer_data_ok(jev)
        return jev
    if purpose == "qa" or provider == ProviderName.INCLOUD_LLM:
        return build_llm_provider(cfg, purpose=purpose, llm=llm)
    if provider == ProviderName.RULES:
        return RuleProvider()
    if classifier is not None:
        if not classifier.is_trained:
            raise ValueError("provider=classifier：传进来的路由分类器还没有训练")
        require_buyer_data_ok(classifier)
        return classifier
    if classifier_samples is None:
        raise ValueError(
            "provider=classifier 需要路由分类器：模型文件（RODIT_CLASSIFIER_PATH）"
            "或训练数据（classifier_samples）"
        )
    return ClassifierProvider(fallback=RuleProvider()).fit(classifier_samples, seed_examples=True)

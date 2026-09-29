"""Jev 判断层 JevProvider（方案 §6.2、§6.3、§6.4、§6.5）。

§6.2 结论：买家消息不能发给 Jev，Jev 暂不进入生产链路。所以：
- purpose="production" 只有 §6.3 的四个条件全部满足时才能构造（否则 ComplianceError），
  这时 accepts_buyer_data=True；配置里还要 DecisionConfig.jev_production_approved=True。
- purpose="offline_eval"（§6.3 "离线评测：用我们自己编写的合成问题"）：accepts_buyer_data=False，
  Router / QualityGate 会拒绝它；decide() 只接受 state["synthetic"] is True、
  且 state 里任何字符串都不含个人信息（core.pii.contains_pii）的请求。
- 请求和响应原样使用 Jev 格式（/api/v1/systemone，§6.5），model 换成 Jev 的模型名；
  响应按 core.decision 的模型校验，再 validate_response。
- 这里没有 HTTP 代码、没有密钥、没有真实地址：transport 由调用方注入（dict → dict）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from rodit.core.decision import DecisionRequest, DecisionResponse, validate_response
from rodit.core.errors import ComplianceError, DecisionError
from rodit.core.pii import contains_pii
from rodit.decision._common import iter_strings

JevPurpose = Literal["production", "offline_eval"]
JevTransport = Callable[[dict[str, Any]], dict[str, Any]]


class JevConditions(BaseModel):
    """§6.3 重新引入生产链路的四个条件（需要同时满足；默认都不满足）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    in_cloud_deployment: bool = False  # 1. 境内部署或私有化部署，能部署在平台云内
    operator_and_dpa: bool = False  # 2. 公开运营主体，数据处理协议写明保留期限、不用于训练
    platform_approved: bool = False  # 3. 通过平台的报备或审核
    chinese_eval_passed: bool = False  # 4. 中文测试集上准确率和概率校准达到境内方案的水平

    @property
    def all_met(self) -> bool:
        return (
            self.in_cloud_deployment
            and self.operator_and_dpa
            and self.platform_approved
            and self.chinese_eval_passed
        )

    def unmet(self) -> list[str]:
        return [name for name, value in self.model_dump().items() if not value]


class JevProvider:
    """Jev 判断层（实现 core.decision.DecisionProvider）。"""

    name = "jev"

    def __init__(
        self,
        transport: JevTransport,
        conditions: JevConditions,
        *,
        purpose: JevPurpose,
        model: str = "jev-1.13.0",
    ) -> None:
        if purpose == "production" and not conditions.all_met:
            raise ComplianceError(
                f"Jev 不能进入生产链路（方案 §6.2、§6.3），未满足的条件：{conditions.unmet()}"
            )
        self.transport = transport
        self.conditions = conditions
        self.purpose: JevPurpose = purpose
        self.model = model

    @property
    def accepts_buyer_data(self) -> bool:
        return self.purpose == "production" and self.conditions.all_met

    def _guard(self, request: DecisionRequest) -> None:
        if self.accepts_buyer_data:
            return
        if request.state.get("synthetic") is not True:
            raise ComplianceError(
                "Jev 离线评测只能用我们编写的合成问题（方案 §6.3）：state.synthetic 必须为 True"
            )
        if any(contains_pii(s) for s in iter_strings(request.state)):
            raise ComplianceError("Jev 离线评测的 state 里有个人信息，拒绝发送（方案 §6.2、§6.3）")

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self._guard(request)
        payload = request.model_dump(mode="json")
        payload["model"] = self.model
        try:
            raw = self.transport(payload)
        except ComplianceError:
            raise
        except Exception as e:  # 注入的网络层可能抛任何异常：统一按"判断层不可用"降级（§12.2）
            raise DecisionError(f"Jev 调用失败：{type(e).__name__}") from e
        if not isinstance(raw, dict):
            raise DecisionError("Jev 响应不是 JSON 对象")
        try:
            response = DecisionResponse.model_validate(raw)
        except ValidationError as e:
            raise DecisionError(f"Jev 响应格式不对：{e.error_count()} 处错误") from e
        validate_response(request, response)
        return response

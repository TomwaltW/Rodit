"""组装与应用服务：把各模块接成一个平台单元（§9.1），并提供应用层流程（§4.2、§5.8、§7.3、§8）。

- unit.build_unit：通用组装（不读环境变量）；
- demo.build_demo_unit：本地演示（合成数据、离线替身），python -m rodit.app.demo 跑 §11 的流程；
- production.build_production_unit：生产部署（PostgreSQL、境内大模型；
  平台工具和适配器由平台云里的实现注入）；
- service.AssistService：一条买家消息、客服反馈、商家操作、售后审核的应用流程。

负责：集成阶段（TΩ）。可以依赖所有子包。
"""

from rodit.app.service import AssistService
from rodit.app.unit import Repos, Unit, build_unit, in_memory_repos

__all__ = ["AssistService", "Repos", "Unit", "build_unit", "in_memory_repos"]

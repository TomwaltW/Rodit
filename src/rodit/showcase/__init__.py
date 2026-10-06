"""客户演示站（python -m rodit.showcase）：用合成数据和离线大模型替身，在浏览器里展示方案的设计——
产品概览、智能辅助工作台（推荐卡片 + 懂→找→写→判→兜底的过程）、售后审核助手、数据看板、
合规与部署。

只用于演示：独立的 FastAPI 应用（create_showcase_app），不挂进生产的 create_app / rodit.ops；
所有店铺、买家、订单都是合成的（CLAUDE.md 只用合成数据），不联网、不需要密钥。
"""

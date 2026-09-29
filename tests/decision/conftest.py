"""判断层测试共用的数据：手写的合成买家问句（带标注的问题类型）。

全部为合成数据（CLAUDE.md）：不含真实买家、订单、手机号或地址。用于规则判断层的准确率检查
和分类器的训练 / 预测测试。INTENT_EXAMPLES（§5.1 例句）另外单独检查。
"""

from __future__ import annotations

import pytest

from rodit.core.enums import Intent

I = Intent  # noqa: E741

SYNTHETIC_INTENT_SAMPLES: tuple[tuple[str, Intent], ...] = (
    # 商品参数
    ("这件是纯棉的吗", I.PRODUCT_SPEC),
    ("面料是什么呀，会不会起球", I.PRODUCT_SPEC),
    ("这个裙子透不透", I.PRODUCT_SPEC),
    ("冬天穿厚吗，有没有加绒", I.PRODUCT_SPEC),
    ("电池容量多少毫安的", I.PRODUCT_SPEC),
    # 尺码
    ("我160，100斤穿什么码", I.SIZE_ADVICE),
    ("平时穿M，这款要拍大一码吗", I.SIZE_ADVICE),
    ("身高170体重120斤选多大", I.SIZE_ADVICE),
    ("这个码数偏小吗", I.SIZE_ADVICE),
    ("腰围72穿几码合适", I.SIZE_ADVICE),
    # 使用方法
    ("这个怎么用啊", I.USAGE),
    ("能不能机洗", I.USAGE),
    ("真丝的要怎么洗", I.USAGE),
    ("有安装说明书吗", I.USAGE),
    # 库存
    ("黑色M码还有吗", I.STOCK),
    ("白色的没货了吗", I.STOCK),
    ("什么时候补货呀", I.STOCK),
    ("L码有现货吗", I.STOCK),
    # 价格优惠
    ("能不能再便宜一点", I.PRICE_PROMO),
    ("有优惠券吗", I.PRICE_PROMO),
    ("现在有什么活动吗", I.PRICE_PROMO),
    ("两件包邮吗", I.PRICE_PROMO),
    ("多少钱能拿", I.PRICE_PROMO),
    # 发货与快递
    ("今天下单今天能发吗", I.SHIPPING_POLICY),
    ("默认发什么快递", I.SHIPPING_POLICY),
    ("可以发顺丰吗", I.SHIPPING_POLICY),
    ("西藏能寄吗", I.SHIPPING_POLICY),
    ("拍了多久发货", I.SHIPPING_POLICY),
    # 发票 / 赠品 / 保障
    ("可以开专票吗", I.INVOICE_GIFT),
    ("买了送运费险吗", I.INVOICE_GIFT),
    ("有赠品吗", I.INVOICE_GIFT),
    ("保修多久", I.INVOICE_GIFT),
    # 对比 / 真伪
    ("你家是正品吗", I.COMPARE_AUTHENTIC),
    ("这款和那款有什么区别", I.COMPARE_AUTHENTIC),
    ("会不会是假货", I.COMPARE_AUTHENTIC),
    ("跟专柜的一样吗", I.COMPARE_AUTHENTIC),
    # 物流进度
    ("快递到哪里了", I.LOGISTICS_TRACK),
    ("物流三天没更新了", I.LOGISTICS_TRACK),
    ("大概几天能到", I.LOGISTICS_TRACK),
    ("帮我查一下物流", I.LOGISTICS_TRACK),
    # 催发货
    ("拍了三天了怎么还不发", I.URGE_SHIP),
    ("发货了吗", I.URGE_SHIP),
    ("麻烦催一下发货", I.URGE_SHIP),
    ("还没发货吗，急用", I.URGE_SHIP),
    # 改单
    ("地址写错了能改吗", I.MODIFY_ORDER),
    ("帮我换成L码", I.MODIFY_ORDER),
    ("我要取消订单", I.MODIFY_ORDER),
    ("拍错颜色了，改成白色", I.MODIFY_ORDER),
    # 退换政策
    ("支持七天无理由吗", I.RETURN_POLICY),
    ("不合适可以退货吗", I.RETURN_POLICY),
    ("退货运费谁承担", I.RETURN_POLICY),
    ("能换货吗", I.RETURN_POLICY),
    # 退款进度
    ("退款怎么还没到账", I.REFUND_PROGRESS),
    ("钱什么时候退回来", I.REFUND_PROGRESS),
    ("退款进度怎么样了", I.REFUND_PROGRESS),
    ("退款多久能到", I.REFUND_PROGRESS),
    # 质量 / 破损 / 错漏发
    ("衣服有个破洞", I.QUALITY_ISSUE),
    ("少发了一件", I.QUALITY_ISSUE),
    ("发错颜色了", I.QUALITY_ISSUE),
    ("收到有很大的异味", I.QUALITY_ISSUE),
    ("杯子碎了", I.QUALITY_ISSUE),
    # 签收未收到
    ("显示已签收可是我没拿到", I.NOT_RECEIVED),
    ("物流说签收了，我根本没收到", I.NOT_RECEIVED),
    ("被别人签收了", I.NOT_RECEIVED),
    ("快递丢了吧", I.NOT_RECEIVED),
    # 使用故障
    ("充不进电", I.MALFUNCTION),
    ("用了两天就不能用了", I.MALFUNCTION),
    ("按了没反应", I.MALFUNCTION),
    ("蓝牙连不上", I.MALFUNCTION),
    # 投诉
    ("我要去12315投诉你们", I.COMPLAINT),
    ("申请平台介入", I.COMPLAINT),
    ("你们这是虚假宣传", I.COMPLAINT),
    ("我要举报这家店", I.COMPLAINT),
    # 问候
    ("你好", I.GREETING),
    ("有人吗", I.GREETING),
    ("亲，在不在", I.GREETING),
    ("怎么不回消息", I.GREETING),
    ("好的谢谢", I.GREETING),
    # 站外交易
    ("加我微信聊", I.OFF_PLATFORM),
    ("支付宝转你行不行", I.OFF_PLATFORM),
    ("有没有qq", I.OFF_PLATFORM),
    ("私下交易便宜点", I.OFF_PLATFORM),
    # 其他
    ("你们公司在哪个城市", I.OTHER),
    ("今天天气不错", I.OTHER),
    ("我考虑一下", I.OTHER),
)
"""合成的带标注买家问句（约 85 条，覆盖全部 20 个问题类型）。"""


@pytest.fixture
def synthetic_intent_samples() -> tuple[tuple[str, Intent], ...]:
    return SYNTHETIC_INTENT_SAMPLES

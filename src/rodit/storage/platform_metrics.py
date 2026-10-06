"""平台客服考核指标的 PostgreSQL 仓储（core.ports.PlatformMetricsRepository；方案 §12.4、§9.2、
迁移版本 6 的 platform_metric 表）。

与内存实现（rodit.inmemory.InMemoryPlatformMetricsRepository）语义一致，共用
rodit.testing.contracts 的 check_platform_metrics_repository / 串店检查。约定同 repositories.py：
- 每个方法开一个 Database.session(shop_id)：行级安全按店铺限定，SQL 里也显式写 WHERE shop_id；
- 读出的时间转成东八区；排序与内存实现一致（日期、指标、来源升序，文本按码点 COLLATE "C"）；
- recorded_by 写入前去掉 NUL 等控制字符（TEXT 不接受 NUL；core.models.strip_control_chars）；
- 数据库异常原样抛出（StorageUnavailable / psycopg.Error），由调用方降级；
- purge_shop 分批删、每批一个短事务（Database.purge_rows，§8.4）。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from rodit.core.clock import CST
from rodit.core.enums import PlatformMetric
from rodit.core.models import PlatformMetricSample, strip_control_chars
from rodit.storage.db import Database

TABLE = "platform_metric"
_FIELDS = (
    "shop_id",
    "metric",
    "day",
    "value",
    "source",
    "sample_size",
    "recorded_by",
    "recorded_at",
)
_COLS = ", ".join(_FIELDS)


def _sample(row: dict[str, Any]) -> PlatformMetricSample:
    return PlatformMetricSample.model_validate(
        {**row, "recorded_at": row["recorded_at"].astimezone(CST)}
    )


class PgPlatformMetricsRepository:
    """实现 core.ports.PlatformMetricsRepository（platform_metric）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(self, sample: PlatformMetricSample) -> PlatformMetricSample:
        recorded_by = (
            strip_control_chars(sample.recorded_by) if sample.recorded_by is not None else None
        )
        placeholders = ", ".join(["%s"] * len(_FIELDS))
        with self._db.session(sample.shop_id) as conn:
            row = conn.execute(
                f"INSERT INTO {TABLE} ({_COLS}) VALUES ({placeholders})"
                " ON CONFLICT (shop_id, metric, day, source) DO UPDATE SET"
                " value = EXCLUDED.value, sample_size = EXCLUDED.sample_size,"
                " recorded_by = EXCLUDED.recorded_by, recorded_at = EXCLUDED.recorded_at"
                f" RETURNING {_COLS}",
                (
                    sample.shop_id,
                    sample.metric.value,
                    sample.day,
                    sample.value,
                    sample.source.value,
                    sample.sample_size,
                    recorded_by,
                    sample.recorded_at,
                ),
            ).fetchone()
        assert row is not None
        return _sample(row)

    def list(
        self,
        shop_id: str,
        *,
        metric: PlatformMetric | None = None,
        since: date | None = None,
        until: date | None = None,
    ) -> list[PlatformMetricSample]:
        conds, params = ["shop_id = %s"], [shop_id]
        if metric is not None:
            conds.append("metric = %s")
            params.append(PlatformMetric(metric).value)
        for clause, value in (("day >= %s", since), ("day <= %s", until)):
            if value is not None:
                conds.append(clause)
                params.append(value)
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_COLS} FROM {TABLE} WHERE {' AND '.join(conds)}"
                ' ORDER BY day, metric COLLATE "C", source COLLATE "C"',
                params,
            ).fetchall()
        return [_sample(r) for r in rows]

    def purge_shop(self, shop_id: str) -> int:
        # §8.4 分批删、每批一个短事务（Database.purge_rows）
        return self._db.purge_rows(shop_id, TABLE)


__all__ = ["PgPlatformMetricsRepository"]

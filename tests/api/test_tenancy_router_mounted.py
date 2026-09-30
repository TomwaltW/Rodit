"""店铺生命周期的路由（rodit.api.routes_tenancy）挂在 create_app 上，应用服务上有 tenancy。"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.testclient import TestClient

from rodit.api import create_app, routes_tenancy
from rodit.api.auth import StaticTokenAuthenticator
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.app.tenancy import TenancyService
from rodit.core.enums import Platform


def test_create_app_mounts_the_tenancy_router(monkeypatch):
    probe = APIRouter()

    @probe.get("/v1/_tenancy_probe")
    def _probe() -> dict[str, str]:
        return {"ok": "yes"}

    monkeypatch.setattr(routes_tenancy, "router", probe)
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    app = create_app(service, authenticator=StaticTokenAuthenticator({}))
    assert "/v1/_tenancy_probe" in app.openapi()["paths"]
    assert TestClient(app).get("/v1/_tenancy_probe").json() == {"ok": "yes"}
    assert isinstance(service.tenancy, TenancyService)


def test_stub_router_is_empty_but_importable():
    assert isinstance(routes_tenancy.router, APIRouter)

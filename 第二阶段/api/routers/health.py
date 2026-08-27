"""轻量服务存活检查。"""

from fastapi import APIRouter

from 第二阶段.api.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


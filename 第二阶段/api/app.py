"""FastAPI 应用工厂与应用级资源生命周期。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from 第二阶段.api.dependencies import ApplicationContainer, build_container
from 第二阶段.api.exception_handlers import register_exception_handlers
from 第二阶段.api.routers import documents, health, questions, sessions


def create_app(container: ApplicationContainer | None = None) -> FastAPI:
    resolved_container = container or build_container()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        resolved_container.close()

    application = FastAPI(
        title=resolved_container.config.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.container = resolved_container
    if resolved_container.config.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_container.config.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
        )
    register_exception_handlers(application)
    application.include_router(health.router)
    application.include_router(sessions.router)
    application.include_router(documents.router)
    application.include_router(questions.router)
    return application


app = create_app()


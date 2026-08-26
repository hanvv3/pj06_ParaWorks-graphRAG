from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.agent_runtime.checkpointing import (
    CheckpointRuntime,
    build_checkpoint_runtime,
)
from backend.app.api.v1.router import api_router
from backend.app.core.config import Settings, get_settings

CheckpointRuntimeFactory = Callable[[Settings], CheckpointRuntime]


def create_app(
    *,
    checkpoint_runtime_factory: CheckpointRuntimeFactory = build_checkpoint_runtime,
) -> FastAPI:
    settings = get_settings()
    checkpoint_runtime = checkpoint_runtime_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        checkpoint_runtime.start()
        app.state.agent_checkpoint_runtime = checkpoint_runtime
        try:
            yield
        finally:
            checkpoint_runtime.close()

    app = FastAPI(title='ParaWorks Harness', lifespan=lifespan)

    @app.get('/health')
    def health() -> dict[str, bool | str]:
        return {'status': 'ok', 'service': 'paraworks', 'demo_mode': settings.paraworks_demo_mode}

    app.include_router(api_router)
    return app


app = create_app()

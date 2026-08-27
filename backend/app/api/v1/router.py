from fastapi import APIRouter, Depends

from backend.app.api.v1 import (
    admin,
    agent_runs,
    ask,
    assistant,
    auth,
    dashboard,
    documents,
    integrations,
    knowledge,
    messages,
    notifications,
    orchestration,
    orchestration_v2,
    projects,
    rag,
    review,
    search,
    stream,
    todos,
)
from backend.app.core.session_auth import check_csrf

api_router = APIRouter(prefix='/api/v1', dependencies=[Depends(check_csrf)])
api_router.include_router(admin.router)
api_router.include_router(agent_runs.router)
api_router.include_router(assistant.router)
api_router.include_router(ask.router)
api_router.include_router(auth.router)
api_router.include_router(dashboard.router)
api_router.include_router(documents.router)
api_router.include_router(integrations.router)
api_router.include_router(knowledge.router)
api_router.include_router(messages.router)
api_router.include_router(notifications.router)
api_router.include_router(orchestration.router)
api_router.include_router(orchestration_v2.router)
api_router.include_router(projects.router)
api_router.include_router(rag.router)
api_router.include_router(review.router)
api_router.include_router(search.router)
api_router.include_router(stream.router)
api_router.include_router(todos.router)

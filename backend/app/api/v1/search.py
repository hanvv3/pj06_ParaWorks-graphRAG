from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from backend.app.api.v1.rag_delivery import (
    RAG_ERROR_RESPONSES,
    RagValidationRoute,
    deliver_direct_rag,
)
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.schemas.rag import SearchV1Projection
from backend.app.schemas.search import SearchRequest

router = APIRouter(prefix='/search', tags=['search'], route_class=RagValidationRoute)
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]


@router.post('', response_model=SearchV1Projection, responses=RAG_ERROR_RESPONSES)
def search_knowledge(
    request: SearchRequest,
    user: CurrentUser,
    http_request: Request,
) -> JSONResponse:
    result = http_request.app.state.rag_application_facade.invoke_search(
        actor=user,
        caller_text=request.query,
    )
    return deliver_direct_rag(result)

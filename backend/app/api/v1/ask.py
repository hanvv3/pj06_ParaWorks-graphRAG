from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from backend.app.api.v1.rag_delivery import (
    RAG_ERROR_RESPONSES,
    RagValidationRoute,
    deliver_direct_rag,
)
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.schemas.ask import AskRequest
from backend.app.schemas.rag import AskV1Projection

router = APIRouter(prefix='/ask', tags=['ask'], route_class=RagValidationRoute)
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]


@router.post('', response_model=AskV1Projection, responses=RAG_ERROR_RESPONSES)
def ask_company_memory(
    request: AskRequest,
    user: CurrentUser,
    http_request: Request,
) -> JSONResponse:
    result = http_request.app.state.rag_application_facade.invoke_ask(
        actor=user,
        caller_text=request.question,
    )
    return deliver_direct_rag(result)

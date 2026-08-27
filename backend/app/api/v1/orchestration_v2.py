from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.responses import JSONResponse

from backend.app.agent_runtime.review_v2_service import (
    ReviewWorkflowService,
    ReviewWorkflowServiceError,
    ReviewWorkflowStatus,
)
from backend.app.core.demo_auth import DemoUser, get_demo_user
from backend.app.schemas.review_workflow import (
    ReviewWorkflowDiagnosticResponse,
    ReviewWorkflowDryRunResponse,
    ReviewWorkflowRunRequest,
    ReviewWorkflowStatusResponse,
)


class _BoundedValidationRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def bounded_route_handler(request: Request):
            try:
                return await original_route_handler(request)
            except RequestValidationError:
                return JSONResponse(
                    status_code=400,
                    content={'detail': {'code': 'invalid_input'}},
                )

        return bounded_route_handler


router = APIRouter(
    prefix='/orchestration/v2/company-memory',
    tags=['orchestration-v2'],
    route_class=_BoundedValidationRoute,
)
CurrentUser = Annotated[DemoUser, Depends(get_demo_user)]
RequestBody = Annotated[object, Body()]

_CONFLICT_CODES = frozenset({
    'idempotency_key_reused',
    'evidence_changed',
    'review_unresolved',
    'runtime_version_unavailable',
    'budget_exceeded',
    'concurrent_resume',
    'invalid_state_transition',
})
_UNAVAILABLE_CODES = frozenset({
    'checkpoint_unavailable',
    'checkpoint_failed',
    'model_unavailable',
})


@router.get('', response_model=ReviewWorkflowDiagnosticResponse)
def diagnostic(
    request: Request,
    _user: CurrentUser,
) -> ReviewWorkflowDiagnosticResponse:
    return _call_service(_workflow_service(request).diagnostic)


@router.post('/dry-run', response_model=ReviewWorkflowDryRunResponse)
def dry_run(
    request: Request,
    body: RequestBody,
    user: CurrentUser,
) -> ReviewWorkflowDryRunResponse:
    run_request = _validated_run_request(body)
    return _call_service(
        _workflow_service(request).dry_run,
        actor=user,
        request=run_request,
    )


@router.post('/runs', response_model=ReviewWorkflowStatusResponse)
def start_run(
    request: Request,
    response: Response,
    body: RequestBody,
    user: CurrentUser,
) -> ReviewWorkflowStatusResponse:
    run_request = _validated_run_request(body)
    status = _call_service(
        _workflow_service(request).start,
        actor=user,
        request=run_request,
    )
    response.status_code = 202 if status.status == 'awaiting_human_review' else 200
    return _status_response(status)


@router.get('/runs/{thread_id}', response_model=ReviewWorkflowStatusResponse)
def run_status(
    thread_id: str,
    request: Request,
    user: CurrentUser,
) -> ReviewWorkflowStatusResponse:
    status = _call_service(
        _workflow_service(request).status,
        actor=user,
        thread_id=thread_id,
    )
    return _status_response(status)


@router.post(
    '/runs/{thread_id}/resume',
    response_model=ReviewWorkflowStatusResponse,
)
def resume_run(
    thread_id: str,
    request: Request,
    user: CurrentUser,
) -> ReviewWorkflowStatusResponse:
    status = _call_service(
        _workflow_service(request).resume,
        actor=user,
        thread_id=thread_id,
    )
    return _status_response(status)


@router.post(
    '/runs/{thread_id}/cancel',
    response_model=ReviewWorkflowStatusResponse,
)
def cancel_run(
    thread_id: str,
    request: Request,
    user: CurrentUser,
) -> ReviewWorkflowStatusResponse:
    status = _call_service(
        _workflow_service(request).cancel,
        actor=user,
        thread_id=thread_id,
    )
    return _status_response(status)


def _workflow_service(request: Request) -> ReviewWorkflowService:
    return request.app.state.review_workflow_service


def _validated_run_request(body: object) -> ReviewWorkflowRunRequest:
    try:
        return ReviewWorkflowRunRequest.model_validate(body)
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail={'code': 'invalid_input'},
        ) from None


def _call_service(call, **kwargs):
    try:
        return call(**kwargs)
    except ReviewWorkflowServiceError as exc:
        public_code = 'not_found' if exc.code == 'permission_denied' else exc.code
        if public_code == 'invalid_input':
            status_code = 400
        elif public_code == 'not_found':
            status_code = 404
        elif public_code in _CONFLICT_CODES:
            status_code = 409
        elif public_code in _UNAVAILABLE_CODES:
            status_code = 503
        else:
            status_code = 409
        raise HTTPException(
            status_code=status_code,
            detail={'code': public_code},
        ) from None


def _status_response(status: ReviewWorkflowStatus) -> ReviewWorkflowStatusResponse:
    return ReviewWorkflowStatusResponse.model_validate(asdict(status))

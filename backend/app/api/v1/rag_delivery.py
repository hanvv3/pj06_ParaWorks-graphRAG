"""HTTP mapping only: no dependencies may be opened after facade delivery."""

import json

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.responses import Response

from backend.app.agent_runtime.rag_application import DirectRagDeliveryResult
from backend.app.schemas.rag import RagPublicErrorResponse, RagValidationErrorResponse

RAG_ERROR_RESPONSES = {
    status: {'model': RagPublicErrorResponse} for status in (403, 409, 500, 502, 503)
}
RAG_ERROR_RESPONSES[422] = {
    'model': RagPublicErrorResponse | RagValidationErrorResponse,
}


class RagValidationRoute(APIRoute):
    """Preserve FastAPI errors even when rejected input contains surrogates."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handle(request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                return Response(
                    status_code=422,
                    media_type='application/json',
                    content=json.dumps(
                        jsonable_encoder({'detail': exc.errors()}),
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(',', ':'),
                    ),
                )

        return handle


def deliver_direct_rag(result: DirectRagDeliveryResult) -> JSONResponse:
    if result.error is not None:
        return JSONResponse(
            status_code=result.public_status,
            content={'detail': {'code': result.error.code}},
        )
    return JSONResponse(
        status_code=200,
        content=result.projection.model_dump(mode='json'),
    )

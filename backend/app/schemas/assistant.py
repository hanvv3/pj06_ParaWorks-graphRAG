from pydantic import BaseModel, Field, field_validator

from backend.app.agent_runtime.rag_v2_identity import StrictUnicodeScalarValidator
from backend.app.schemas.rag import ExactV1Projection, RagCitationResponse


class AssistantConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=160)

    @field_validator('title')
    @classmethod
    def validate_title_unicode(cls, value: str | None) -> str | None:
        if value is not None:
            StrictUnicodeScalarValidator.validate(value)
        return value


class AssistantMessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)

    @field_validator('content')
    @classmethod
    def validate_content_unicode(cls, value: str) -> str:
        return StrictUnicodeScalarValidator.validate(value)


class AssistantConversationResponse(BaseModel):
    id: int
    title: str
    summary: str | None
    created_at: str
    updated_at: str


class AssistantMessageResponse(ExactV1Projection):
    id: int
    conversation_id: int
    role: str
    content: str
    citations: list[RagCitationResponse]
    source_ids: list[str]
    source_links: list[str]
    source_snippets: list[str]
    permission_level: str | None
    hidden_match_count: int
    permission_notice: str | None
    agent_run_id: int | None
    metadata: dict[str, object]
    created_at: str


class AssistantConversationsResponse(BaseModel):
    conversations: list[AssistantConversationResponse]


class AssistantConversationCreatedResponse(BaseModel):
    conversation: AssistantConversationResponse


class AssistantMessagesResponse(BaseModel):
    conversation: AssistantConversationResponse
    messages: list[AssistantMessageResponse]


class AssistantTurnResponse(BaseModel):
    conversation: AssistantConversationResponse
    user_message: AssistantMessageResponse
    assistant_message: AssistantMessageResponse


class AssistantEmailSendResponse(BaseModel):
    message: AssistantMessageResponse
    status: str
    gmail_message_id: str | None = None

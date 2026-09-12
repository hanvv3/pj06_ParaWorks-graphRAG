from pydantic import BaseModel, Field, field_validator

from backend.app.agent_runtime.rag_v2_identity import StrictUnicodeScalarValidator


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator('question')
    @classmethod
    def validate_unicode(cls, value: str) -> str:
        return StrictUnicodeScalarValidator.validate(value)

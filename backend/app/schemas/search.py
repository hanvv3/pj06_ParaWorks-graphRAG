from pydantic import BaseModel, Field, field_validator

from backend.app.agent_runtime.rag_v2_identity import StrictUnicodeScalarValidator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator('query')
    @classmethod
    def validate_unicode(cls, value: str) -> str:
        return StrictUnicodeScalarValidator.validate(value)

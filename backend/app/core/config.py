import logging
import os
from decimal import Decimal
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _coerce_integer_literal(value: object) -> object:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    paraworks_env: str = 'local'
    paraworks_demo_mode: bool = False
    paraworks_seed_demo_data: bool = False
    paraworks_database_url: str | None = None
    paraworks_demo_database_url: str | None = None
    langgraph_review_v2_enabled: bool = False
    langgraph_rag_v2_enabled: bool = False
    langgraph_strict_msgpack: bool = False
    langgraph_checkpoint_retention_days: int = Field(default=30, ge=1, le=3650)
    agent_runtime_fingerprint_secret: str = (
        'local-development-agent-runtime-fingerprint-secret'
    )
    agent_runtime_fingerprint_key_version: str = 'v1'
    auto_review_mode: Literal['disabled', 'shadow', 'enforce'] = 'disabled'
    auto_review_enforce_percentage: Annotated[
        Literal[0, 10, 100], BeforeValidator(_coerce_integer_literal)
    ] = 0
    auto_review_launch_confirmation_ttl_seconds: int = Field(
        default=600, ge=60, le=3600
    )
    auto_review_provider_timeout_seconds: int = Field(default=60, ge=10, le=90)
    auto_review_provider_send_start_window_seconds: int = Field(
        default=5, ge=1, le=10
    )
    auto_review_provider_attempt_lease_seconds: int = Field(
        default=120, ge=60, le=600
    )
    auto_review_provider_commit_grace_seconds: int = Field(
        default=30, ge=10, le=120
    )
    auto_review_validator_input_cost_per_1m_tokens: Decimal | None = Field(
        default=None, gt=0
    )
    auto_review_validator_output_cost_per_1m_tokens: Decimal | None = Field(
        default=None, gt=0
    )
    auto_review_extraction_input_cost_per_1m_tokens: Decimal | None = Field(
        default=None, gt=0
    )
    auto_review_extraction_output_cost_per_1m_tokens: Decimal | None = Field(
        default=None, gt=0
    )
    auto_review_max_total_cost_usd: Decimal = Field(
        default=Decimal('0.20'), gt=0, le=Decimal('0.20')
    )
    agent_runtime_security_scope_id: str = 'default'
    auth_session_cookie_name: str = 'paraworks_session'
    auth_refresh_cookie_name: str = 'paraworks_refresh'
    auth_session_secret: str = 'local-development-session-secret'
    auth_session_ttl_seconds: int = 60 * 60 * 24 * 30
    auth_refresh_ttl_seconds: int = 60 * 60 * 24 * 30
    auth_cookie_secure: bool = False
    auth_csrf_cookie_name: str = 'paraworks_csrf'
    auth_csrf_header_name: str = 'X-CSRF-Token'
    database_url: str = 'postgresql+psycopg://paraworks:paraworks@localhost:5432/paraworks'
    redis_url: str = 'redis://localhost:6379/0'
    celery_task_always_eager: bool = True
    openai_api_key: str | None = None
    openai_embedding_model: str = 'text-embedding-3-small'
    openai_embedding_dimensions: int = 1536
    openai_embedding_timeout_seconds: float = 30.0
    openai_embedding_input_cost_per_1m_tokens: float = 0.02
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    agent_llm_enabled: bool = False
    agent_llm_provider_order: str = 'openai,gemini'
    agent_llm_openai_primary_model: str = 'gpt-5.4'
    agent_llm_openai_model: str = 'gpt-5.4-mini'
    agent_llm_gemini_model: str = 'gemini-2.5-flash'
    agent_llm_input_cost_per_1m_tokens: float = 0.15
    agent_llm_output_cost_per_1m_tokens: float = 0.60
    agent_llm_max_estimated_cost_usd: float | None = 0.001
    agent_llm_max_input_chars: int = 12000
    agent_llm_max_evidence_messages: int = 12
    agent_llm_max_output_tokens: int = 512
    agent_llm_temperature: float = 0.2
    agent_llm_timeout_seconds: float = 30.0
    assistant_email_agent_enabled: bool = True
    assistant_email_agent_model: str = 'gpt-4.1-nano'
    assistant_email_draft_agent_model: str = 'gpt-5.4-mini'
    assistant_email_agent_max_input_chars: int = 2500
    assistant_email_agent_max_output_tokens: int = 256
    assistant_email_agent_temperature: float = 0.0
    assistant_email_agent_timeout_seconds: float = 12.0
    assistant_email_agent_min_confidence: float = 0.72
    rag_embedding_max_estimated_cost_usd: float | None = 0.001
    rag_use_pgvector_search: bool = False
    slack_bot_token: str | None = None
    slack_user_token: str | None = None
    slack_channel_ids: str = ''
    slack_workspace_url: str = 'https://3aiagent.slack.com'
    slack_client_id: str | None = None
    slack_client_secret: str | None = None
    slack_oauth_redirect_uri: str = 'http://localhost:3000/integrations/slack/callback'
    slack_oauth_state_secret: str = 'local-development-state-secret'
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_oauth_redirect_uri: str = 'http://localhost:3000/integrations/google/callback'
    google_oauth_state_secret: str = 'local-development-google-state-secret'
    google_identity_redirect_uri: str = 'http://localhost:3000/login/google/callback'
    google_identity_state_secret: str = 'local-development-google-identity-state-secret'
    google_drive_sync_enabled: bool = True
    google_drive_sync_interval_seconds: int = 3600  # Default 1 hour
    gmail_sync_enabled: bool = True
    gmail_sync_interval_seconds: int = 10  # For testing, recommended 3600 for production

    @model_validator(mode='after')
    def _validate_auto_review_profile(self) -> 'Settings':
        from backend.app.agent_runtime.auto_review_cost_policy import (
            AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS,
            AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS,
            AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS,
            AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS,
        )

        if self.auto_review_mode in {'disabled', 'shadow'}:
            if self.auto_review_enforce_percentage != 0:
                raise ValueError('disabled and shadow modes require enforce percentage 0')
        elif self.auto_review_enforce_percentage not in {10, 100}:
            raise ValueError('enforce mode requires percentage 10 or 100')
        if self.auto_review_provider_attempt_lease_seconds <= (
            self.auto_review_provider_send_start_window_seconds
            + self.auto_review_provider_timeout_seconds
            + self.auto_review_provider_commit_grace_seconds
        ):
            raise ValueError('provider attempt lease must exceed send window, timeout, and grace')
        if (
            self.auto_review_provider_timeout_seconds,
            self.auto_review_provider_send_start_window_seconds,
            self.auto_review_provider_attempt_lease_seconds,
            self.auto_review_provider_commit_grace_seconds,
        ) != (
            AUTO_REVIEW_PROVIDER_TIMEOUT_SECONDS,
            AUTO_REVIEW_PROVIDER_SEND_START_WINDOW_SECONDS,
            AUTO_REVIEW_PROVIDER_ATTEMPT_LEASE_SECONDS,
            AUTO_REVIEW_PROVIDER_COMMIT_GRACE_SECONDS,
        ):
            raise ValueError('provider timing values must match the immutable cost policy')
        if (
            self.auto_review_mode != 'disabled'
            and self.agent_runtime_fingerprint_secret
            == 'local-development-agent-runtime-fingerprint-secret'
        ):
            raise ValueError('non-disabled auto review requires a non-local fingerprint secret')
        return self

    def require_auto_review_live_readiness(
        self,
        *,
        model: object | None = None,
        http_hook: object | None = None,
    ) -> None:
        """Reject an unsafe paid C.5 configuration before provider admission."""
        if self.auto_review_mode == 'disabled':
            return
        self.require_c5_durable_key_ready()
        from langchain_core.globals import get_debug

        from backend.app.agent_runtime.auto_review_cost_policy import (
            confirmation_prices_match_registry,
            is_server_owned_fenced_send_hook,
        )

        if not confirmation_prices_match_registry(
            extraction_input=self.auto_review_extraction_input_cost_per_1m_tokens,
            extraction_output=self.auto_review_extraction_output_cost_per_1m_tokens,
            validation_input=self.auto_review_validator_input_cost_per_1m_tokens,
            validation_output=self.auto_review_validator_output_cost_per_1m_tokens,
        ):
            raise ValueError('auto-review confirmation prices must match the registry')
        if not self.openai_api_key:
            raise ValueError('non-disabled auto review requires an OpenAI key')
        if model is not None and bool(getattr(model, 'verbose', False)):
            raise ValueError('non-disabled auto review requires verbose models disabled')
        if http_hook is not None and not is_server_owned_fenced_send_hook(http_hook):
            raise ValueError('non-disabled auto review rejects an unapproved HTTP hook')
        if get_debug() or os.getenv('OPENAI_LOG', '').lower() == 'debug':
            raise ValueError('non-disabled auto review requires debug logging disabled')
        if logging.getLogger('openai').getEffectiveLevel() <= logging.DEBUG:
            raise ValueError('non-disabled auto review requires the OpenAI logger above DEBUG')

    def require_c5_durable_key_ready(self) -> None:
        """Fail closed before any durable C.5 keyed row or PostgreSQL bootstrap."""
        secret = self.agent_runtime_fingerprint_secret
        if (
            not self.agent_runtime_fingerprint_key_version.strip()
            or secret == 'local-development-agent-runtime-fingerprint-secret'
            or len(secret.encode('utf-8')) < 32
        ):
            raise ValueError('durable C.5 key requires a non-placeholder 32-byte secret and key version')

    def allows_c5_process_local_sqlite_smoke(self) -> bool:
        """Only the disabled in-memory smoke path may retain the local placeholder."""
        if self.auto_review_mode != 'disabled':
            return False
        from sqlalchemy.engine import make_url

        try:
            url = make_url(self.resolved_database_url())
        except Exception:
            return False
        return url.get_backend_name() == 'sqlite' and url.database in {None, ':memory:'}

    def resolved_database_url(self) -> str:
        if self.paraworks_demo_mode and self.paraworks_demo_database_url:
            return self.paraworks_demo_database_url
        if not self.paraworks_demo_mode and self.paraworks_database_url:
            return self.paraworks_database_url
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    def resolved_database_url(self) -> str:
        if self.paraworks_demo_mode and self.paraworks_demo_database_url:
            return self.paraworks_demo_database_url
        if not self.paraworks_demo_mode and self.paraworks_database_url:
            return self.paraworks_database_url
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()

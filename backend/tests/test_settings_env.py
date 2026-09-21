from pathlib import Path

import pytest

from backend.app.core.config import Settings


def test_template_documents_optional_graph_cache_and_opt_in_consumers(monkeypatch):
    from dotenv import dotenv_values

    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    template = Path(__file__).parents[2] / '.env.example'
    values = dotenv_values(template)
    settings = Settings(_env_file=template)
    for name in ('RAG_GRAPH_ENRICHMENT_ENABLED', 'RAG_ANSWER_CACHE_ENABLED',
                 'ASSISTANT_EMAIL_AGENT_ENABLED', 'GOOGLE_DRIVE_SYNC_ENABLED',
                 'GMAIL_SYNC_ENABLED'):
        assert values[name] == 'false'
        assert getattr(settings, name.lower()) is False
    for name in ('RAG_NEO4J_URI', 'RAG_NEO4J_USERNAME', 'RAG_NEO4J_PASSWORD'):
        assert values[name] == ''
    assert values['OPENAI_API_KEY'] == ''


def test_tracked_env_template_loads_with_safe_disabled_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('AUTO_REVIEW_MODE', raising=False)
    monkeypatch.delenv('AUTO_REVIEW_ENFORCE_PERCENTAGE', raising=False)
    monkeypatch.delenv('AGENT_LLM_ENABLED', raising=False)
    template_path = Path(__file__).parents[2] / '.env.example'

    settings = Settings(_env_file=template_path)

    assert settings.auto_review_mode == 'disabled'
    assert settings.auto_review_enforce_percentage == 0
    assert settings.agent_llm_enabled is False


@pytest.mark.parametrize(
    ('raw_percentage', 'expected_percentage'),
    [('0', 0), ('10', 10), ('100', 100)],
)
def test_settings_loads_auto_review_percentage_from_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_percentage: str,
    expected_percentage: int,
) -> None:
    monkeypatch.delenv('AUTO_REVIEW_MODE', raising=False)
    monkeypatch.delenv('AUTO_REVIEW_ENFORCE_PERCENTAGE', raising=False)
    mode = 'disabled' if raw_percentage == '0' else 'enforce'
    env_file = tmp_path / '.env'
    env_file.write_text(
        '\n'.join(
            [
                f'AUTO_REVIEW_MODE={mode}',
                f'AUTO_REVIEW_ENFORCE_PERCENTAGE={raw_percentage}',
                'AGENT_RUNTIME_FINGERPRINT_SECRET=' + ('x' * 32),
            ]
        ),
        encoding='utf-8',
    )

    settings = Settings(_env_file=env_file)

    assert settings.auto_review_enforce_percentage == expected_percentage

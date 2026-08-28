import pytest

from backend.app.admin.data_reset import reset_connector_derived_data
from backend.app.core.config import Settings
from backend.app.models import (
    AgentRun,
    AssistantConversation,
    AuthUser,
    AutoReviewRuntimeKeyState,
    Document,
    DocumentChunk,
    DocumentVersion,
    IntegrationConnection,
    Source,
)


def _seed_reset_rows(db_session):
    source = Source(
        source_type='gmail',
        source_id='gmail-reset',
        source_url='https://gmail.mock/reset',
        title='Reset source',
        permission_level='internal',
        raw_metadata={},
    )
    db_session.add(source)
    db_session.flush()
    document = Document(source_id=source.id, title='Reset document')
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version='v1', body='Reset body')
    db_session.add(version)
    db_session.flush()
    db_session.add(
        DocumentChunk(
            version_id=version.id,
            source_id=source.id,
            chunk_index=0,
            text='Reset body',
            source_snippet='Reset body',
            permission_level='internal',
            metadata_={},
        )
    )
    db_session.add(
        AgentRun(
            agent_name='rag_orchestrator_agent',
            prompt_version='rag-answer:v1',
            status='complete',
            source_window='ask:reset',
            cache_key='reset',
            model_name='fake-rag-orchestrator-model',
            permission_level='internal',
            metadata_={},
        )
    )
    db_session.add(AssistantConversation(user_id='demo-admin', title='Reset conversation'))
    db_session.add(
        AuthUser(
            external_id='reset-user',
            email='reset@example.com',
            display_name='Reset User',
            role='admin',
            department='Platform',
            title='Admin',
            permission_levels=['internal'],
        )
    )
    db_session.add(
        IntegrationConnection(
            connector_type='gmail',
            workspace_id='acct-reset',
            workspace_name='reset@example.com',
            token_ref='gmail:acct-reset',
            masked_bot_token='token...reset',
            scopes=['gmail.readonly'],
        )
    )
    db_session.commit()


def test_reset_connector_derived_data_dry_run_does_not_delete(db_session) -> None:
    _seed_reset_rows(db_session)

    result = reset_connector_derived_data(
        db_session,
        settings=Settings(paraworks_env='local'),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.deleted_counts['sources'] == 1
    assert db_session.query(Source).count() == 1
    assert db_session.query(AuthUser).count() == 1


def test_reset_connector_derived_data_requires_local_and_confirm(db_session) -> None:
    _seed_reset_rows(db_session)

    with pytest.raises(ValueError, match='confirm=True'):
        reset_connector_derived_data(
            db_session,
            settings=Settings(paraworks_env='local'),
            dry_run=False,
            confirm=False,
        )
    with pytest.raises(ValueError, match='local environment'):
        reset_connector_derived_data(
            db_session,
            settings=Settings(paraworks_env='production'),
            dry_run=False,
            confirm=True,
        )


def test_reset_connector_derived_data_deletes_sources_but_preserves_auth_and_integrations(db_session) -> None:
    _seed_reset_rows(db_session)

    result = reset_connector_derived_data(
        db_session,
        settings=Settings(paraworks_env='local'),
        dry_run=False,
        confirm=True,
    )

    assert result.dry_run is False
    assert db_session.query(Source).count() == 0
    assert db_session.query(Document).count() == 0
    assert db_session.query(AgentRun).count() == 0
    assert db_session.query(AssistantConversation).count() == 0
    assert db_session.query(AuthUser).count() == 1
    assert db_session.query(IntegrationConnection).count() == 1


def test_reset_connector_derived_data_refuses_retained_c5_state(db_session) -> None:
    _seed_reset_rows(db_session)
    db_session.add(
        AutoReviewRuntimeKeyState(
            component='auto_review_trust_promotion',
            fingerprint_key_version='v1',
            fingerprint_key_material_verifier='a' * 64,
            generation=1,
            ready=False,
        )
    )
    db_session.commit()

    preview = reset_connector_derived_data(
        db_session,
        settings=Settings(paraworks_env='local'),
        dry_run=True,
    )
    assert preview.retained_c5_state is True

    with pytest.raises(ValueError, match='recreate.*disposable local database'):
        reset_connector_derived_data(
            db_session,
            settings=Settings(paraworks_env='local'),
            dry_run=False,
            confirm=True,
        )

    assert db_session.query(Source).count() == 1
    assert db_session.query(AutoReviewRuntimeKeyState).count() == 1

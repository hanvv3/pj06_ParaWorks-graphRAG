from backend.app.connectors.mock import get_mock_connector
from backend.app.seeds.mock_sources import SEED_EVENTS


def test_seed_events_can_be_imported_directly() -> None:
    assert SEED_EVENTS


def test_mock_drive_connector_returns_permission_leakage_case() -> None:
    connector = get_mock_connector('drive')
    events = connector.fetch_events()

    restricted = next(
        event
        for event in events
        if event.source_id == 'drive:permission-leakage-case'
    )
    assert restricted.permission_level == 'restricted'
    assert restricted.source_url.startswith('https://drive.mock/')


def test_all_mock_connectors_return_source_evidence() -> None:
    for connector_type in ['drive', 'gmail', 'slack', 'calendar']:
        events = get_mock_connector(connector_type).fetch_events()

        assert events
        assert all(event.source_url for event in events)
        assert all(event.body for event in events)


def test_mock_gmail_connector_includes_attachment_boundary_event() -> None:
    events = get_mock_connector('gmail').fetch_events()

    attachment = next(event for event in events if event.source_type == 'gmail_attachment')
    assert (
        attachment.source_id
        == 'gmail_attachment:project-alpha-redis-summary:att-budget-pdf'
    )
    assert (
        attachment.raw_metadata['parent_source_id']
        == 'gmail:project-alpha-redis-summary'
    )
    assert attachment.raw_metadata['parser_name'] == 'gmail_attachment_metadata'
    assert attachment.raw_metadata['parser_status'] == 'metadata_only'
    assert attachment.raw_metadata['parser_status_reason'] == 'pdf_parser_not_enabled'
    assert attachment.raw_metadata['content_signature'].endswith(':2048')


def test_fetched_events_do_not_share_mutable_seed_state() -> None:
    first_event = get_mock_connector('slack').fetch_events()[0]
    first_event.participants.append('mutated@example.com')
    first_event.raw_metadata['mutated'] = True

    later_event = get_mock_connector('slack').fetch_events()[0]

    assert 'mutated@example.com' not in later_event.participants
    assert 'mutated' not in later_event.raw_metadata


def test_project_beta_events_include_scope_trigger() -> None:
    beta_events = [
        event
        for connector_type in ['slack', 'calendar']
        for event in get_mock_connector(connector_type).fetch_events()
        if event.raw_metadata['scenario'] == 'project-beta-scope-cut'
    ]

    assert beta_events
    assert all('scope' in event.body.lower() for event in beta_events)

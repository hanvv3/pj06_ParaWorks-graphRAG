from dataclasses import replace

from backend.app.connectors.base import ConnectorManifest
from backend.app.connectors.google import GOOGLE_CONNECTOR_SCOPES
from backend.app.connectors.mock import MOCK_CONNECTOR_MANIFESTS
from backend.app.connectors.slack import SLACK_REQUIRED_SCOPES


def list_connector_manifests(*, demo_mode: bool = True) -> list[ConnectorManifest]:
    return [
        _manifest_for_mode(MOCK_CONNECTOR_MANIFESTS[connector_type], demo_mode=demo_mode)
        for connector_type in sorted(MOCK_CONNECTOR_MANIFESTS)
    ]


def get_connector_manifest(connector_type: str, *, demo_mode: bool = True) -> ConnectorManifest:
    try:
        return _manifest_for_mode(MOCK_CONNECTOR_MANIFESTS[connector_type], demo_mode=demo_mode)
    except KeyError as exc:
        raise ValueError(f'Unsupported connector: {connector_type}') from exc


def _manifest_for_mode(manifest: ConnectorManifest, *, demo_mode: bool) -> ConnectorManifest:
    if demo_mode:
        return manifest
    live_scopes = GOOGLE_CONNECTOR_SCOPES.get(manifest.connector_type, manifest.required_scopes)
    if manifest.connector_type == 'slack':
        live_scopes = SLACK_REQUIRED_SCOPES
    return replace(manifest, mode='live', required_scopes=live_scopes)

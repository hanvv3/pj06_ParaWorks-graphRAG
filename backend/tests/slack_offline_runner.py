"""Explicit offline harness for Slack fixture checks; never reads local credentials.

Run with ``python -m backend.tests.slack_offline_runner [pytest selectors/options]``.
With no selectors this runs the exact historical SLACK_TEN manifest entries.
"""
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

import httpx
import pytest
from pydantic_settings import BaseSettings


def main(*, postgres_url: str | None = None) -> int:
    # Change settings sources BEFORE importing any application module.
    BaseSettings.settings_customise_sources = classmethod(
        lambda cls, settings_cls, init_settings, env_settings, dotenv_settings,
        file_secret_settings: (init_settings, env_settings)
    )
    from backend.app.core.config import Settings

    setting_names = {name.upper() for name in Settings.model_fields}
    prefixes = ('PARAWORKS_', 'SLACK_', 'GOOGLE_', 'OPENAI_', 'ANTHROPIC_',
                'LANGGRAPH_', 'LANGCHAIN_', 'LANGSMITH_', 'AGENT_RUNTIME_',
                'AUTO_REVIEW_', 'RAG_', 'NEO4J_')
    for name in list(os.environ):
        if name.upper() in setting_names or name.upper().startswith(prefixes):
            os.environ.pop(name)
    os.environ.update(PARAWORKS_DEMO_MODE='true',
                      PARAWORKS_DATABASE_URL='sqlite://',
                      PARAWORKS_DEMO_DATABASE_URL='sqlite://',
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    if postgres_url is not None:
        from sqlalchemy.engine import make_url
        locator = make_url(postgres_url)
        if (locator.get_backend_name() != 'postgresql'
                or locator.host != '127.0.0.1'
                or not (locator.database or '').endswith('_test')
                or not (locator.username or '').endswith('_test')):
            raise ValueError('disposable local PostgreSQL required')
        os.environ['PARAWORKS_TEST_POSTGRES_URL'] = postgres_url
    blocked_calls = []

    def blocked(*args, **kwargs):
        blocked_calls.append('external_transport')
        raise AssertionError('offline Slack harness blocked external transport')

    original_connect = socket.socket.connect
    original_socketpair = socket.socketpair
    socketpair_state = threading.local()

    def local_socketpair(*args, **kwargs):
        socketpair_state.active = True
        try:
            return original_socketpair(*args, **kwargs)
        finally:
            socketpair_state.active = False

    def guarded_connect(sock, address):
        # Windows implements the stdlib's private socketpair via loopback TCP.
        if getattr(socketpair_state, 'active', False):
            return original_connect(sock, address)
        return blocked()

    socket.socketpair = local_socketpair
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    # Keep MockTransport and the in-process TestClient operational.
    httpx.HTTPTransport.handle_request = blocked
    httpx.AsyncHTTPTransport.handle_async_request = blocked
    from backend.tests.release_contracts import SLACK_TEN

    Path('.tmp').mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='slack-s1-', dir='.tmp'))
    args = sys.argv[1:] or [*SLACK_TEN, '-q', '--tb=short']
    result = pytest.main([*args, f'--basetemp={run_dir / "temp"}',
                          '-o', f'cache_dir={run_dir / "cache"}'])
    print(f'Offline transport attempts: {len(blocked_calls)}')
    return result if not blocked_calls else 1


if __name__ == '__main__':
    raise SystemExit(main())

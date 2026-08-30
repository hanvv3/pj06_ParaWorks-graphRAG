from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal


class ReleaseBootstrapRefused(RuntimeError):  # noqa: N818 - frozen release contract
    code = 'environment_refused'


@dataclass(frozen=True, slots=True)
class EffectiveBootstrapState:
    resolved_database_backend: Literal['sqlite']
    dotenv_source_enabled: Literal[False]
    demo_mode: Literal[True]
    auto_review_mode: Literal['disabled']
    agent_llm_enabled: Literal[False]
    live_provider_credentials_present: Literal[False]
    paid_provider_flags_present: Literal[False]
    pytest_worker_count: Literal[1]


def apply_test_environment_guard() -> None:
    if os.getenv('PYTEST_XDIST_WORKER') or '-n' in os.getenv('PYTEST_ADDOPTS', '').split():
        raise ReleaseBootstrapRefused('parallel_pytest_refused')
    os.environ['PARAWORKS_DEMO_MODE'] = 'true'
    os.environ['AUTO_REVIEW_MODE'] = 'disabled'
    os.environ.pop('AUTO_REVIEW_ENFORCE_PERCENTAGE', None)
    os.environ['AGENT_LLM_ENABLED'] = 'false'
    for name in tuple(os.environ):
        upper = name.upper()
        if any(marker in upper for marker in ('OPENAI_API_KEY', 'SLACK_TOKEN', 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'ALLOW_PAID')):
            os.environ.pop(name, None)
    from backend.app.core.config import Settings, get_settings

    if get_settings.cache_info().currsize:
        raise ReleaseBootstrapRefused('settings_cache_prepopulated')
    Settings.model_config['env_file'] = None
    get_settings.cache_clear()


def probe_effective_state() -> EffectiveBootstrapState:
    apply_test_environment_guard()
    database_url = os.environ.get('DATABASE_URL', '')
    if not database_url.startswith('sqlite'):
        raise ReleaseBootstrapRefused('sqlite_locator_required')
    return EffectiveBootstrapState('sqlite', False, True, 'disabled', False, False, False, 1)


def initialize_guarded_sqlite() -> None:
    probe_effective_state()
    from sqlalchemy import create_engine

    from backend.app.core.config import Settings
    from backend.app.db.base import Base

    settings = Settings(_env_file=None)
    engine = create_engine(settings.resolved_database_url())
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = tuple(argv or ())
    try:
        if args == ('probe',):
            print(json.dumps(asdict(probe_effective_state()), sort_keys=True))
        elif args == ('init-sqlite',):
            initialize_guarded_sqlite()
            print('{"initialized":true}')
        else:
            raise ReleaseBootstrapRefused('command_refused')
    except ReleaseBootstrapRefused:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main(os.sys.argv[1:]))

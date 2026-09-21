"""Behavior checks for local launch tooling; no live DB or provider clients."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def runtime():
    try:
        return importlib.import_module('scripts.local_runtime')
    except ModuleNotFoundError:
        pytest.fail('shared safe local runtime is missing')


def test_environment_process_wins_and_normal_start_disables_demo(tmp_path):
    (tmp_path / '.env').write_text(
        'OPENAI_API_KEY=file-secret\nDATABASE_URL=sqlite:///file.db\n'
        'PARAWORKS_DEMO_MODE=true\nPARAWORKS_SEED_DEMO_DATA=true\n',
        encoding='utf-8',
    )
    env = runtime().load_environment(tmp_path, {'OPENAI_API_KEY': 'process-secret'})
    assert env['OPENAI_API_KEY'] == 'process-secret'
    assert env['DATABASE_URL'] == 'sqlite:///file.db'
    assert env['PARAWORKS_DEMO_MODE'] == 'false'
    assert env['PARAWORKS_SEED_DEMO_DATA'] == 'false'
    assert 'AGENT_LLM_ENABLED' not in env
    assert (tmp_path / '.env').read_text().startswith('OPENAI_API_KEY=file-secret')


def test_native_failure_is_sanitized_and_propagated(tmp_path, capsys):
    module = runtime()
    with pytest.raises(module.OperationRefused, match='native_command_failed'):
        module.run_checked(
            [sys.executable, '-c', 'print("secret-value"); raise SystemExit(17)'],
            workspace=tmp_path,
            env=dict(os.environ),
        )
    assert 'secret-value' not in capsys.readouterr().out


def test_provider_preflight_never_constructs_network_clients(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(
        socket, 'create_connection', lambda *a, **k: pytest.fail('network attempted')
    )
    module = runtime()
    from backend.app.core.config import Settings

    settings = Settings(
        _env_file=None, agent_llm_enabled=False, openai_api_key='secret-value'
    )
    result = module.provider_preflight(settings)
    assert result['paid_calls'] == 0
    assert result['live_allowed'] is False
    assert result['agent_llm_enabled'] is False
    assert 'secret-value' not in json.dumps(result)


def test_live_metadata_probe_is_one_bounded_authenticated_request(monkeypatch):
    import httpx
    from openai import OpenAI

    from backend.app.core.config import Settings

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'gpt-5.4-mini',
                'object': 'model',
                'created': 0,
                'owned_by': 'openai',
            },
        )

    def client_factory(**kwargs):
        assert kwargs['max_retries'] == 0
        assert kwargs['timeout'] == 10.0
        return OpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))
        )

    result = runtime().provider_preflight(
        Settings(_env_file=None, openai_api_key='fake-key'),
        live=True,
        client_factory=client_factory,
    )
    assert result['connectivity'] == 'ok'
    assert result['inference_calls'] == 0
    assert len(requests) == 1
    assert requests[0].method == 'GET'
    assert requests[0].url.path == '/v1/models/gpt-5.4-mini'
    assert requests[0].headers['authorization'] == 'Bearer fake-key'
    assert 'fake-key' not in json.dumps(result)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process ownership')
def test_owned_child_stop_and_pid_reuse_refusal(tmp_path):
    module = runtime()
    child = module.spawn_owned(
        workspace=tmp_path,
        name='fixture',
        command=[sys.executable, '-c', 'import time; time.sleep(120)'],
        env=dict(os.environ),
    )
    try:
        assert module.ownership_status(child, tmp_path) == 'running'
        wrong = dict(child, created='not-the-same-process')
        with pytest.raises(module.OperationRefused, match='process_ownership_mismatch'):
            module.stop_owned(wrong, tmp_path)
        assert module.ownership_status(child, tmp_path) == 'running'
        assert 'command' not in child
        module.stop_owned(child, tmp_path)
        assert module.ownership_status(child, tmp_path) == 'stopped'
    finally:
        if module.ownership_status(child, tmp_path) == 'running':
            module.stop_owned(child, tmp_path)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process ownership')
def test_partial_startup_cleans_only_owned_processes(tmp_path):
    module = runtime()
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    captured = []
    try:

        def fail_readiness(records):
            captured.extend(records)
            raise module.OperationRefused('readiness_failed')

        with pytest.raises(module.OperationRefused, match='readiness_failed'):
            module.start_group(
                tmp_path,
                [('fixture', [sys.executable, '-c', 'import time; time.sleep(120)'])],
                dict(os.environ),
                fail_readiness,
            )
        assert foreign.poll() is None
        assert all(
            module.ownership_status(item, tmp_path) == 'stopped' for item in captured
        )
        assert not (tmp_path / '.tmp/local-runtime/state.json').exists()
    finally:
        foreign.terminate()
        foreign.wait(timeout=10)


def test_occupied_port_refuses_without_killing_listener():
    import socket

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(runtime().OperationRefused, match='port_in_use'):
            runtime().require_free_ports('127.0.0.1', [port])
        assert listener.getsockname()[1] == port


@pytest.mark.skipif(os.name != 'nt', reason='Windows process ownership')
def test_stop_reclaims_supervisor_descendants(tmp_path):
    import time

    module = runtime()
    receipt = tmp_path / 'child.pid'
    script = f'import os,time,pathlib; pathlib.Path({str(receipt)!r}).write_text(str(os.getpid())); time.sleep(120)'
    record = module.spawn_owned(
        workspace=tmp_path,
        name='tree',
        command=[sys.executable, '-c', script],
        env=dict(os.environ),
    )
    child_pid = None
    try:
        for _ in range(50):
            if receipt.exists():
                break
            time.sleep(0.1)
        child_pid = int(receipt.read_text())
        assert module.process_identity(child_pid) is not None
        module.stop_owned(record, tmp_path)
        assert module.process_identity(child_pid) is None
    finally:
        if module.ownership_status(record, tmp_path) == 'running':
            module.stop_owned(record, tmp_path)
        if child_pid and module.process_identity(child_pid):
            subprocess.run(
                ['taskkill', '/PID', str(child_pid), '/F'], capture_output=True
            )


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell wrappers')
def test_powershell_wrapper_preflight_and_failure_exit(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[2]
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('test-provider.ps1', 'local-runtime-common.ps1', 'local_runtime.py'):
        shutil.copy(root / 'scripts' / name, scripts / name)
    # Select the real module from the project while using an isolated dotenv root.
    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        OPENAI_API_KEY='canary-secret',
        AGENT_LLM_ENABLED='false',
    )
    (tmp_path / '.env').write_text('AGENT_LLM_OPENAI_MODEL=gpt-5.4-mini\n')
    command = [
        'powershell.exe',
        '-NoProfile',
        '-NonInteractive',
        '-File',
        str(scripts / 'test-provider.ps1'),
        '-PythonPath',
        sys.executable,
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['connectivity'] == 'not_tested'
    assert 'canary-secret' not in result.stdout + result.stderr
    env['AUTO_REVIEW_ENFORCE_PERCENTAGE'] = '999'
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'canary-secret' not in result.stdout + result.stderr


def test_legacy_demo_refuses_without_touching_database(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/run_e2e_demo.py')],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=dict(os.environ, DATABASE_URL='sqlite:///must-not-create.db'),
    )
    assert result.returncode == 2
    assert 'retired' in result.stderr
    assert not (tmp_path / 'must-not-create.db').exists()


def test_sync_cli_defaults_to_no_write_and_redacts_failures(monkeypatch, capsys):
    from scripts import sync_slack

    def fails():
        raise RuntimeError('provider-secret-canary')

    monkeypatch.setattr(sync_slack, 'run_sync', fails)
    assert sync_slack.main([]) == 2
    assert sync_slack.main(['--execute']) == 1
    output = capsys.readouterr()
    assert 'provider-secret-canary' not in output.out + output.err


def test_service_diagnostics_refuse_sqlite_without_creating_file(tmp_path):
    from backend.app.core.config import Settings

    database = tmp_path / 'do-not-create.db'
    settings = Settings(
        _env_file=None,
        paraworks_demo_mode=False,
        paraworks_database_url=f'sqlite:///{database}',
    )
    with pytest.raises(runtime().OperationRefused, match='postgresql_required'):
        runtime().check_services(settings)
    assert not database.exists()


def test_child_log_redaction_removes_urls_keys_and_passwords():
    env = {
        'DATABASE_URL': 'postgresql://person:pass-canary@localhost/db',
        'OPENAI_API_KEY': 'key-canary',
        'RAG_NEO4J_PASSWORD': 'neo-canary',
    }
    text = runtime().redact_output(
        'bad key-canary pass-canary neo-canary postgresql://person:pass-canary@localhost/db',
        env,
    )
    assert 'canary' not in text


@pytest.mark.skipif(os.name != 'nt', reason='Windows process ownership')
def test_group_stop_refuses_foreign_member_before_stopping_any_owned(tmp_path):
    module = runtime()
    state = module.start_group(
        tmp_path,
        [('one', [sys.executable, '-c', 'import time; time.sleep(120)'])],
        dict(os.environ),
        lambda records: None,
    )
    record = state['processes'][0]
    try:
        state['processes'].append(dict(record, created='reused-pid'))
        module.save_state(tmp_path, state)
        with pytest.raises(module.OperationRefused, match='process_ownership_mismatch'):
            module.stop_group(tmp_path)
        assert module.ownership_status(record, tmp_path) == 'running'
    finally:
        module.stop_owned(record, tmp_path)


def test_changed_owner_receipt_is_refused(tmp_path):
    if os.name != 'nt':
        pytest.skip('Windows process ownership')
    module = runtime()
    record = module.spawn_owned(
        workspace=tmp_path,
        name='owner',
        command=[sys.executable, '-c', 'import time; time.sleep(120)'],
        env=dict(os.environ),
    )
    try:
        assert (
            module.ownership_status(dict(record, owner='forged'), tmp_path) == 'foreign'
        )
    finally:
        module.stop_owned(record, tmp_path)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process ownership')
def test_restart_replaces_owned_process_and_reloads_config_preserving_ports(
    tmp_path, monkeypatch
):
    module = runtime()
    commands = [('fixture', [sys.executable, '-c', 'import time; time.sleep(120)'])]
    launch = {
        'host': '127.0.0.1',
        'backend_port': 18531,
        'frontend_port': 18532,
        'skip_frontend': True,
        'smoke_database': None,
    }
    state = module.start_group(
        tmp_path, commands, dict(os.environ), lambda records: None, launch=launch
    )
    old = state['processes'][0]
    (tmp_path / '.env').write_text('LOCAL_RESTART_CANARY=fresh-config\n')
    original_env, original_cwd = dict(os.environ), Path.cwd()
    started = []

    def fixture_start(args, workspace, env):
        assert args.backend_port == 18531
        assert args.frontend_port == 18532
        assert args.skip_frontend is True
        assert env['LOCAL_RESTART_CANARY'] == 'fresh-config'
        assert module.ownership_status(old, workspace) == 'stopped'
        replacement = module.start_group(
            workspace, commands, env, lambda records: None, launch=launch
        )
        started.extend(replacement['processes'])
        return {'status': 'running'}

    monkeypatch.setattr(module, 'start_application', fixture_start)
    try:
        assert module.main(['restart', '--workspace', str(tmp_path)]) == 0
        assert len(started) == 1
        assert started[0]['pid'] != old['pid']
        assert module.ownership_status(started[0], tmp_path) == 'running'
    finally:
        module.stop_group(tmp_path)
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)


def test_skip_app_checks_services_even_when_app_port_is_occupied(tmp_path, monkeypatch):
    import socket
    from argparse import Namespace

    module = runtime()
    # Only the external DB diagnostic is replaced; port refusal remains real.
    monkeypatch.setattr(module, 'check_services', lambda settings: {'postgresql': 'ok'})
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        args = Namespace(
            host='127.0.0.1',
            backend_port=port,
            frontend_port=port,
            skip_frontend=True,
            skip_app=True,
            smoke_database=None,
            initialize_database=False,
        )
        env = {
            'PARAWORKS_DEMO_MODE': 'false',
            'PARAWORKS_DATABASE_URL': 'postgresql+psycopg://fake:fake@127.0.0.1/test',
        }
        assert module.start_application(args, tmp_path, env) == {
            'status': 'services_ready'
        }
        assert listener.getsockname()[1] == port
    assert not (tmp_path / '.tmp/local-runtime/state.json').exists()


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell wrapper behavior')
@pytest.mark.parametrize('native_failure', [False, True])
def test_visual_wrapper_blocks_backend_seed_and_restores_environment(
    tmp_path, native_failure
):
    import shutil

    root = Path(__file__).resolve().parents[2]
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (tmp_path / 'frontend').mkdir()
    shutil.copy(root / 'scripts/run-visual-smoke.ps1', scripts / 'run-visual-smoke.ps1')
    (scripts / 'start-smoke.ps1').write_text("Write-Output 'fixture-started'\n")
    (scripts / 'stop.ps1').write_text("Write-Output 'fixture-stopped'\n")
    driver = tmp_path / 'run-fixture.ps1'
    driver.write_text(
        "$env:PLAYWRIGHT_SKIP_BACKEND_SEED = 'original-value'\n"
        'function global:npm.cmd {\n'
        "    Write-Output ('suite-seed=' + $env:PLAYWRIGHT_SKIP_BACKEND_SEED)\n"
        f'    $global:LASTEXITCODE = {17 if native_failure else 0}\n'
        '}\n'
        "try { & (Join-Path $PSScriptRoot 'scripts/run-visual-smoke.ps1') }\n"
        "catch { Write-Output 'fixture-native-failure' }\n"
        "Write-Output ('restored-seed=' + $env:PLAYWRIGHT_SKIP_BACKEND_SEED)\n",
        encoding='utf-8',
    )
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(driver)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'suite-seed=1' in result.stdout
    assert 'restored-seed=original-value' in result.stdout
    assert 'fixture-stopped' in result.stdout
    assert ('fixture-native-failure' in result.stdout) is native_failure

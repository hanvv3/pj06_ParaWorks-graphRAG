"""Small Windows local-process manager and read-only service diagnostics.

Root .env is read in-process, with inherited environment taking precedence.
Never persists configuration or credentials. State is an ownership receipt, not
authority to kill a PID without rechecking its creation time and command hash.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class OperationRefused(RuntimeError):  # noqa: N818 - bounded CLI refusal
    """Only bounded, non-sensitive diagnostic codes may cross the CLI boundary."""


def load_environment(workspace: Path, inherited=None, *, smoke_database=None):
    from dotenv import dotenv_values

    values = dotenv_values(workspace / '.env', interpolate=False)
    env = {name.upper(): value for name, value in values.items() if value is not None}
    env.update(
        {
            name.upper(): value
            for name, value in (os.environ if inherited is None else inherited).items()
        }
    )
    env['PARAWORKS_DEMO_MODE'] = 'true' if smoke_database else 'false'
    env['PARAWORKS_SEED_DEMO_DATA'] = 'true' if smoke_database else 'false'
    if smoke_database:
        env['PARAWORKS_DEMO_DATABASE_URL'] = f'sqlite:///{smoke_database}'
        env['DATABASE_URL'] = env['PARAWORKS_DEMO_DATABASE_URL']
        # Smoke must never consume inherited live connector or model credentials.
        for name in list(env):
            if name.endswith(('_API_KEY', '_BOT_TOKEN', '_USER_TOKEN')):
                env[name] = ''
        env.update(
            AGENT_LLM_ENABLED='false',
            AUTO_REVIEW_MODE='disabled',
            AUTO_REVIEW_ENFORCE_PERCENTAGE='0',
            LANGGRAPH_RAG_V2_ENABLED='false',
            LANGGRAPH_RAG_V2_MODE='disabled',
            RAG_GRAPH_ENRICHMENT_ENABLED='false',
            RAG_ANSWER_CACHE_ENABLED='false',
            RAG_USE_PGVECTOR_SEARCH='false',
            RAG_RETRIEVAL_BACKEND='keyword',
        )
    return env


def settings_from_environment(env):
    from backend.app.core.config import Settings

    # Passing explicit values prevents an unrelated cwd dotenv from taking over.
    fields = {
        name: env[name.upper()] for name in Settings.model_fields if name.upper() in env
    }
    return Settings(_env_file=None, **fields)


def run_checked(command, *, workspace, env):
    result = subprocess.run(
        command, cwd=workspace, env=env, capture_output=True, text=True
    )
    if result.returncode:
        raise OperationRefused('native_command_failed')
    return result


def redact_output(message, env):
    from urllib.parse import unquote, urlsplit

    sensitive = set()
    for name, value in env.items():
        if not value:
            continue
        if any(word in name for word in ('SECRET', 'TOKEN', 'PASSWORD', 'API_KEY')):
            sensitive.add(value)
        if 'URL' in name or 'URI' in name:
            try:
                password = urlsplit(value).password
                if password:
                    sensitive.update((value, password, unquote(password)))
            except ValueError:
                sensitive.add(value)
    for value in sorted(sensitive, key=len, reverse=True):
        message = message.replace(value, '[REDACTED]')
    return message


def provider_preflight(settings, *, live=False, client_factory=None):
    report = {
        'mode': 'metadata_connectivity' if live else 'configuration_only',
        'agent_llm_enabled': settings.agent_llm_enabled,
        'openai_key_present': bool(settings.openai_api_key),
        'gemini_key_present': bool(settings.gemini_api_key or settings.google_api_key),
        'paid_calls': 0,
        'inference_calls': 0,
        'live_allowed': False,
        'workflow_readiness': 'use_authenticated_application_preflight_and_confirmation',
    }
    if not live:
        report['connectivity'] = 'not_tested'
        return report
    if not settings.openai_api_key:
        raise OperationRefused('openai_key_missing')
    model = settings.agent_llm_openai_model
    if (
        not model
        or len(model) > 200
        or any(
            c
            not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._:'
            for c in model
        )
    ):
        raise OperationRefused('openai_model_invalid')
    if client_factory is None:
        from openai import OpenAI

        client_factory = OpenAI
    try:
        with client_factory(
            api_key=settings.openai_api_key,
            timeout=10.0,
            max_retries=0,
            base_url='https://api.openai.com/v1',
        ) as client:
            client.models.retrieve(model)
    except Exception:
        raise OperationRefused('provider_metadata_connection_failed') from None
    report.update(connectivity='ok', metadata_requests=1)
    return report


def check_services(settings, *, include_api=False, host='127.0.0.1', port=8000):
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    from scripts.check_db_schema import check_schema

    if make_url(settings.resolved_database_url()).get_backend_name() != 'postgresql':
        raise OperationRefused('postgresql_required')
    engine = create_engine(
        settings.resolved_database_url(),
        connect_args={'connect_timeout': 5, 'options': '-c statement_timeout=5000'},
    )
    report = {}
    try:
        with engine.connect() as connection:
            connection.execute(text('SELECT 1'))
        schema = check_schema(
            engine, expected_embedding_dimensions=settings.openai_embedding_dimensions
        )
        report['postgresql'] = 'ok'
        report['schema'] = (
            'ok'
            if schema.ok
            else 'missing_or_outdated_run_start_InitializeDatabase_explicitly'
        )
        if not schema.ok:
            raise OperationRefused('schema_missing_or_outdated_use_InitializeDatabase')
    except OperationRefused:
        raise
    except Exception:
        raise OperationRefused('postgresql_connection_or_schema_check_failed') from None
    finally:
        engine.dispose()
    if settings.rag_graph_enrichment_enabled:
        from neo4j import GraphDatabase

        try:
            with (
                GraphDatabase.driver(
                    settings.rag_neo4j_uri,
                    auth=(settings.rag_neo4j_username, settings.rag_neo4j_password),
                    connection_timeout=5,
                    connection_acquisition_timeout=5,
                ) as driver,
                driver.session(database=settings.rag_neo4j_database) as session,
            ):
                from neo4j import Query

                session.run(Query('RETURN 1', timeout=5)).consume()
            report['neo4j'] = 'ok'
        except Exception:
            raise OperationRefused('neo4j_connection_failed') from None
    else:
        report['neo4j'] = 'not_required_graph_disabled'
    if not settings.celery_task_always_eager:
        import redis

        try:
            with redis.Redis.from_url(
                settings.redis_url, socket_connect_timeout=5, socket_timeout=5
            ) as client:
                client.ping()
            report['redis'] = 'ok'
        except Exception:
            raise OperationRefused('redis_connection_failed') from None
    else:
        report['redis'] = 'not_required_eager_tasks'
    if include_api:
        import httpx

        try:
            response = httpx.get(
                f'http://{host}:{port}/health', timeout=5, trust_env=False
            )
            response.raise_for_status()
            if response.json().get('status') != 'ok':
                raise ValueError
            report['api'] = 'ok'
        except Exception:
            raise OperationRefused('api_health_failed') from None
    return report


def require_windows():
    if os.name != 'nt':
        raise OperationRefused('process_lifecycle_requires_windows_powershell')


def process_identity(pid):
    require_windows()
    if not isinstance(pid, int) or pid <= 0:
        raise OperationRefused('invalid_state')
    command = (
        f'$p = Get-CimInstance Win32_Process -Filter "ProcessId = {pid}"; '
        'if ($p) { @{created=$p.CreationDate.ToUniversalTime().ToString("o"); '
        'command=$p.CommandLine; executable=$p.ExecutablePath} | ConvertTo-Json -Compress }'
    )
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise OperationRefused('process_identity_unavailable')
    if not result.stdout.strip():
        return None
    raw = json.loads(result.stdout)
    if not raw['command'] or not raw['executable']:
        raise OperationRefused('process_identity_unavailable')
    return {
        'created': raw['created'],
        'command_hash': hashlib.sha256(raw['command'].encode()).hexdigest(),
        'executable': raw['executable'],
        'command': raw['command'],
    }


def ownership_status(record, workspace):
    if record.get('workspace') != str(workspace.resolve()) or not record.get('owner'):
        return 'foreign'
    current = process_identity(record['pid'])
    if current is None:
        return 'stopped'
    command = current.pop('command')
    if (
        str(Path(__file__).resolve()) not in command
        or str(workspace.resolve()) not in command
        or f'--owner {record["owner"]}' not in command
        or '_supervise' not in command
    ):
        return 'foreign'
    return (
        'running'
        if all(current[key] == record.get(key) for key in current)
        else 'foreign'
    )


def stop_owned(record, workspace):
    status = ownership_status(record, workspace)
    if status == 'foreign':
        raise OperationRefused('process_ownership_mismatch')
    if status == 'stopped':
        return
    # Open a handle and compare creation time once more before terminating it;
    # a reused numeric PID cannot be killed across the check/terminate boundary.
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000 | 0x0001 | 0x00100000, False, record['pid'])
    if not handle:
        raise OperationRefused('process_handle_unavailable')
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    try:
        if ownership_status(record, workspace) != 'running':
            raise OperationRefused('process_ownership_mismatch')
        if not kernel.TerminateProcess(handle, 0):
            raise OperationRefused('process_stop_failed')
        kernel.WaitForSingleObject(handle, 10000)
    finally:
        kernel.CloseHandle(handle)


def attach_kill_job():
    """An OS job closes all descendants if its supervisor dies or is stopped."""
    require_windows()
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_int64),
            ('PerJobUserTimeLimit', ctypes.c_int64),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class Counters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in (
                'ReadOperationCount',
                'WriteOperationCount',
                'OtherOperationCount',
                'ReadTransferCount',
                'WriteTransferCount',
                'OtherTransferCount',
            )
        ]

    class Extended(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', Basic),
            ('IoInfo', Counters),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    limits = Extended()
    limits.BasicLimitInformation.LimitFlags = 0x2000
    if (
        not job
        or not kernel.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        )
        or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())
    ):
        raise OperationRefused('process_job_setup_failed')
    # Intentionally held until OS process exit. Never inherited by children.
    return job


def state_path(workspace):
    path = workspace / '.tmp' / 'local-runtime'
    path.mkdir(parents=True, exist_ok=True)
    return path / 'state.json'


def read_state(workspace):
    path = state_path(workspace)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['workspace'] != str(workspace.resolve()) or result['version'] != 1:
            raise ValueError
        return result
    except (ValueError, KeyError):
        raise OperationRefused('invalid_state') from None


def save_state(workspace, value):
    target = state_path(workspace)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(target)


@contextlib.contextmanager
def lifecycle_lock(workspace):
    require_windows()
    import msvcrt

    with state_path(workspace).with_suffix('.lock').open('a+b') as file:
        file.seek(0)
        if file.read(1) == b'':
            file.write(b'0')
            file.flush()
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise OperationRefused('lifecycle_operation_already_running') from None
        try:
            yield
        finally:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


def spawn_owned(*, workspace, name, command, env):
    require_windows()
    owner = secrets.token_hex(16)
    directory = state_path(workspace).parent
    # The private pipe carries the command, never secrets in argv/state files.
    wrapper = [
        sys.executable,
        str(Path(__file__).resolve()),
        '_supervise',
        '--workspace',
        str(workspace.resolve()),
        '--owner',
        owner,
    ]
    with (directory / f'{name}.log').open('ab') as log:
        process = subprocess.Popen(
            wrapper,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            process.stdin.write(json.dumps(command).encode())
            process.stdin.close()
            identity = process_identity(process.pid)
            if identity is None or process.poll() is not None:
                raise OperationRefused('process_exited_during_start')
            identity.pop('command')
            return dict(
                identity,
                pid=process.pid,
                workspace=str(workspace.resolve()),
                owner=owner,
                name=name,
            )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            raise


def start_group(workspace, commands, env, wait_ready, *, launch=None):
    existing = read_state(workspace)
    if existing and existing.get('processes'):
        raise OperationRefused('existing_state_use_status_or_stop')
    state = {
        'version': 1,
        'workspace': str(workspace.resolve()),
        'processes': [],
        'launch': launch or {},
    }
    try:
        for name, command in commands:
            state['processes'].append(
                spawn_owned(workspace=workspace, name=name, command=command, env=env)
            )
            save_state(workspace, state)
        wait_ready(state['processes'])
    except BaseException:
        for record in reversed(state['processes']):
            stop_owned(record, workspace)
        state_path(workspace).unlink(missing_ok=True)
        raise
    return state


def stop_group(workspace):
    state = read_state(workspace)
    if not state:
        return {'status': 'not_managed', 'processes': []}
    # Validate the entire group before any stop, including restart's stop phase.
    if any(
        ownership_status(item, workspace) == 'foreign' for item in state['processes']
    ):
        raise OperationRefused('process_ownership_mismatch')
    for record in reversed(state['processes']):
        stop_owned(record, workspace)
    state_path(workspace).unlink(missing_ok=True)
    return {'status': 'stopped'}


def require_free_ports(host, ports):
    if host not in {'127.0.0.1', 'localhost'}:
        raise OperationRefused('local_loopback_host_required')
    if len(ports) != len(set(ports)) or any(not 1 <= port <= 65535 for port in ports):
        raise OperationRefused('invalid_ports')
    for port in ports:
        with socket.socket() as listener:
            if os.name == 'nt':
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                listener.bind((host, port))
            except OSError:
                raise OperationRefused('port_in_use_choose_another_port') from None


def wait_http(url, records, workspace, *, timeout=60):
    import httpx

    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if any(ownership_status(item, workspace) != 'running' for item in records):
                raise OperationRefused('process_exited_during_start')
            time.sleep(0.5)
    raise OperationRefused('startup_health_timeout')


def start_application(args, workspace, env):
    if read_state(workspace):
        raise OperationRefused('existing_state_use_status_or_stop')
    if not args.skip_app:
        ports = [args.backend_port] + (
            [] if args.skip_frontend else [args.frontend_port]
        )
        require_free_ports(args.host, ports)
    settings = settings_from_environment(env)
    if args.smoke_database:
        db_path = Path(args.smoke_database)
        if not db_path.is_absolute():
            db_path = workspace / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [sys.executable, '-m', 'backend.app.db.init_db'],
            workspace=workspace,
            env=env,
        )
    else:
        from sqlalchemy.engine import make_url

        if (
            make_url(settings.resolved_database_url()).get_backend_name()
            != 'postgresql'
        ):
            raise OperationRefused('postgresql_required')
        if args.initialize_database:
            settings.require_c5_durable_key_ready()
            run_checked(
                [sys.executable, '-m', 'alembic', 'upgrade', 'head'],
                workspace=workspace,
                env=env,
            )
            run_checked(
                [sys.executable, '-m', 'backend.app.db.init_db'],
                workspace=workspace,
                env=env,
            )
        check_services(settings)
    if args.skip_app:
        return {'status': 'services_ready'}
    env = dict(
        env,
        NEXT_PUBLIC_API_BASE_URL=f'http://{args.host}:{args.backend_port}',
        NEXT_DIST_DIR='.next-smoke' if args.smoke_database else '.next',
    )
    commands = [
        (
            'backend',
            [
                sys.executable,
                '-m',
                'uvicorn',
                'backend.app.main:app',
                '--host',
                args.host,
                '--port',
                str(args.backend_port),
            ],
        )
    ]
    if not args.skip_frontend:
        node = shutil.which('node')
        next_bin = workspace / 'frontend/node_modules/next/dist/bin/next'
        if not node or not next_bin.is_file():
            raise OperationRefused('frontend_dependencies_missing')
        commands.append(
            (
                'frontend',
                [
                    node,
                    str(next_bin),
                    'dev',
                    str(workspace / 'frontend'),
                    '--hostname',
                    args.host,
                    '--port',
                    str(args.frontend_port),
                ],
            )
        )

    def ready(records):
        wait_http(f'http://{args.host}:{args.backend_port}/health', records, workspace)
        if not args.skip_frontend:
            wait_http(
                f'http://{args.host}:{args.frontend_port}/login', records, workspace
            )

    launch = {
        'host': args.host,
        'backend_port': args.backend_port,
        'frontend_port': args.frontend_port,
        'skip_frontend': args.skip_frontend,
        'smoke_database': args.smoke_database,
    }
    state = start_group(workspace, commands, env, ready, launch=launch)
    return {
        'status': 'running',
        'backend_port': args.backend_port,
        'frontend_port': None if args.skip_frontend else args.frontend_port,
        'processes': [{'name': p['name'], 'pid': p['pid']} for p in state['processes']],
        'logs': '.tmp/local-runtime/*.log',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'action',
        choices=[
            'start',
            'stop',
            'restart',
            'status',
            'services',
            'provider',
            'worker',
            '_supervise',
        ],
    )
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--backend-port', type=int, default=8000)
    parser.add_argument('--frontend-port', type=int, default=3000)
    parser.add_argument('--initialize-database', action='store_true')
    parser.add_argument('--skip-frontend', action='store_true')
    parser.add_argument('--skip-app', action='store_true')
    parser.add_argument('--smoke-database')
    parser.add_argument('--include-api', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--owner')
    args = parser.parse_args(argv)
    try:
        workspace = args.workspace.resolve(strict=True)
        if args.action == '_supervise':
            attach_kill_job()
            command = json.loads(sys.stdin.buffer.read())
            child = subprocess.Popen(
                command,
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in child.stdout:
                print(redact_output(line, os.environ), end='', flush=True)
            return child.wait()
        env = load_environment(workspace, smoke_database=args.smoke_database)
        os.environ.clear()
        os.environ.update(env)
        os.chdir(workspace)
        if args.action == 'provider':
            report = provider_preflight(settings_from_environment(env), live=args.live)
        elif args.action == 'services':
            report = check_services(
                settings_from_environment(env),
                include_api=args.include_api,
                host=args.host,
                port=args.backend_port,
            )
        elif args.action == 'worker':
            env['CELERY_TASK_ALWAYS_EAGER'] = 'false'
            check_services(settings_from_environment(env))
            return subprocess.call(
                [
                    sys.executable,
                    '-m',
                    'celery',
                    '-A',
                    'backend.app.tasks.celery_app.celery_app',
                    'worker',
                    '--loglevel=warning',
                    '--pool=solo',
                ],
                env=env,
                cwd=workspace,
            )
        else:
            with lifecycle_lock(workspace):
                if args.action == 'status':
                    state = read_state(workspace)
                    report = (
                        {'status': 'not_managed', 'processes': []}
                        if not state
                        else {
                            'status': 'managed',
                            'processes': [
                                {
                                    'name': item['name'],
                                    'pid': item['pid'],
                                    'status': ownership_status(item, workspace),
                                }
                                for item in state['processes']
                            ],
                            'launch': state['launch'],
                        }
                    )
                elif args.action == 'stop':
                    report = stop_group(workspace)
                else:
                    if args.action == 'restart':
                        state = read_state(workspace)
                        if state:
                            for name, value in state['launch'].items():
                                setattr(args, name, value)
                            env = load_environment(
                                workspace, smoke_database=args.smoke_database
                            )
                        stop_group(workspace)
                    report = start_application(args, workspace, env)
        print(json.dumps(report))
        return 0
    except OperationRefused as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 1
    except Exception:
        print(
            json.dumps({'error': 'local_operation_failed_check_configuration'}),
            file=sys.stderr,
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

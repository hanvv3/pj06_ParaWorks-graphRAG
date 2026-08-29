# Whole-Suite PostgreSQL Isolation Design

검토 버전: 2
작성일: 2026-08-29
상태: 서면 spec 사용자 승인 완료 · 독립 Spec/Quality 검토 PASS · 상세 구현 계획 사용자 검토 대기 · 구현 미승인

관련 문서:

- `plan.md`
- `docs/superpowers/plans/2026-08-28-auto-review-trust-promotion.md`
- `docs/superpowers/plans/2026-08-29-whole-suite-postgresql-isolation.md`
- `docs/superpowers/specs/2026-08-28-auto-review-trust-promotion-design.md`
- `docs/superpowers/specs/2026-08-29-database-storage-initialization-boundary-design.md`
- `.superpowers/sdd/2026-08-29-database-storage-initialization-boundary/final-nonslack-suite-report.md`

## 1. 결정 요약

ParaWorks의 전체 backend release gate는 하나의 controller-owned PostgreSQL
database와 role을 사용하되, PostgreSQL 의존 테스트마다 고유한 schema lease를
사용한다. 일반 애플리케이션 테스트는 별도의 임시 SQLite database에 연결하고,
PostgreSQL URL은 테스트 resource locator로만 전달한다.

이 설계의 핵심 결정은 다음과 같다.

1. `public` schema를 여러 PostgreSQL 테스트 모듈이 공유하지 않는다.
2. 각 모듈 또는 테스트는 `search_path=<owned_schema>,public`인 명시적 URL을
   사용한다.
3. release controller는 애플리케이션 Settings를 바꾸는
   `PARAWORKS_DATABASE_URL`, fingerprint secret, rollout/provider 값을 전체
   pytest process에 주입하지 않는다.
4. PostgreSQL resource locator인 `PARAWORKS_TEST_POSTGRES_URL`과 pgvector
   collection marker인 `PARAWORKS_PGVECTOR_TEST_DATABASE_URL`만 collection
   전에 제공할 수 있다. 각 테스트는 실제 작업 시 schema-leased URL을 사용한다.
5. C.5 evidence trigger를 완화하지 않는다. targeted RED가 가설을 확인한 경우에만
   `test_review_v2_postgres.py`의 legacy fake draft를 실제 C.5 provenance shape로
   보정한다.
6. release pytest는 serial로 실행한다. `xdist`는 허용하지 않는다.
7. Slack 관련 기존 열 개의 유예 항목 외에 새 deselection, skip, expected failure를
   추가하지 않는다.
8. controller와 pytest가 만든 schema, database, role만 정리하며 기존 resource를
   채택하거나 삭제하지 않는다.

이 경계는 C.5 Task 16의 release infrastructure 선행 조건이다. 지금 설계하는
것은 Task 16을 Tasks 6–15보다 먼저 구현한다는 의미가 아니다.

## 2. 관찰된 실패와 원인

최종 non-Slack 비교 실행은 다음 결과를 보존했다.

```text
1453 passed
1 skipped
10 approved Slack deselections
12 failed
144 errors
cleanup 0:0:0:0
```

### 2.1 144개의 setup error

144개는 독립된 product defect가 아니라 두 fixture의 setup refusal이 fan-out된
결과다.

- `test_auto_review_migration.py`: module fixture 1회 실패가 96개 node에 전파됨
- `test_review_v21_extraction_postgres.py`: function fixture 48회가 각각 실패함

앞선 PostgreSQL 모듈이 동일 physical database의 `public` schema에 migration과
checkpoint table을 남긴다. 이후 두 fixture는 고유 schema를 만들기 전에 database
전체의 public table 존재 여부를 검사하여 fail-closed로 중단한다. 두 fixture가 실제
작업에는 이미 고유 `search_path` schema를 사용한다는 점에서, freshness 검사는
database 전체가 아니라 소유한 schema 경계로 이동해야 한다.

### 2.2 여섯 개의 Settings 계약 실패

기존 controller는 다음 값을 전체 pytest process에 주입했다.

```text
PARAWORKS_TEST_POSTGRES_URL=<same postgres database>
PARAWORKS_DATABASE_URL=<same postgres database>
DATABASE_URL=<same postgres database>
AGENT_RUNTIME_FINGERPRINT_SECRET=<non-default test secret>
```

`Settings(_env_file=None)`은 dotenv만 끄며 process environment는 계속 읽는다.
또한 `resolved_database_url()`은 non-demo에서 `paraworks_database_url`을
`database_url`보다 먼저 선택한다. 그 결과 MySQL/SQLite URL이나 local-default
fingerprint를 검증하는 테스트가 controller 환경에 의해 다른 계약을 보게 됐다.

Settings precedence나 보안 검사를 변경하지 않는다. child environment를
정상화하여 이 여섯 테스트가 원래 입력만 보도록 한다.

### 2.3 여섯 개의 Review V2 PostgreSQL 실패와 검증 가설

보존된 전체-suite 출력에는 여섯 node id만 있고 assertion traceback은 남지 않았다.
따라서 아직 관찰된 원인을 확정하지 않는다. 정적 코드 검토로 얻은 우선 검증 가설은
`test_review_v2_postgres.py`의 legacy fake drafting path가 post-C.5 workflow
ReviewItem을 만들면서 다음 provenance를 완성하지 않는다는 것이다.

- 실제 `ReviewItem.agent_run_id`
- `candidate_contract_version='c5-v1'`
- 같은 workflow/evidence를 가리키는 `ReviewItemEvidenceRef`

구현 Task 0에서 clean dedicated schema와 long traceback으로 exact trigger를 먼저
확인한다. 가설이 확인되면 production trigger, trust rule, evidence requirement를
약화하지 않고 테스트 fixture만 실제 C.5 shape로 수정한다. 다른 원인이 관찰되면
fixture를 추측으로 변경하지 않고 해당 owning contract로 되돌린다. 유효한
provenance로 commit한 뒤에도 행동 assertion이 실패할 때만 별도 product TDD
defect로 분류한다.

## 3. 목표와 비목표

### 3.1 목표

- 동일 테스트 집합이 파일/모듈 collection 순서와 무관하게 실행된다.
- PostgreSQL 테스트가 서로의 table, Alembic revision, checkpoint, row를 보지
  못한다.
- 일반 Settings 테스트가 controller의 PostgreSQL/fingerprint 설정을 상속하지
  않는다.
- 현재 `pytest backend/tests`의 단일 collection 의미를 유지한다.
- non-Slack gate는 승인된 Slack 열 개만 제외하고 전부 통과한다.
- full gate는 정확히 기존 Slack 열 개만 실패하고 다른 failure/error가 없다.
- PostgreSQL release-critical 테스트는 skip 없이 실행된다.
- 실패 시에도 owned schema/database/role 정리가 끝나고 `0:0:0:0`이 확인된다.
- provider/connector API를 호출하지 않고 raw source, prompt, secret, DSN을
  출력하지 않는다.

### 3.2 비목표

- product runtime, API, output schema, Alembic product migration 변경
- Review Queue trust/promotion/revoke/duplicate/permission 정책 변경
- token/cost/model/rollout 정책 변경
- LangChain/LangGraph dependency 또는 graph behavior 변경
- CDC/streaming, Deliverable D/E, Neo4j, Slack 복구 작업
- xdist/parallel release execution
- shared Docker container/volume 생성·삭제 정책 변경
- paid Terra/Mini 또는 live provider 평가 실행

## 4. 아키텍처

```text
backend_release_matrix.py
  -> validate shared Postgres container and exact server identity
  -> create one owned _test role + one owned _test database
  -> CREATE EXTENSION vector
  -> create one controller-owned non-lease sentinel schema
  -> sanitize child environment
  -> guarded test bootstrap probes Settings and initializes temporary SQLite
  -> pytest collection and release gates (serial)
       -> ordinary tests use SQLite application DATABASE_URL
       -> PostgreSQL modules receive immutable base resource locator
       -> postgres_schema_lease(...) creates an owned schema
       -> explicit leased URL drives Alembic/session/checkpoint/pgvector
       -> consumers close engines/pools/savers
       -> lease drops only its owned schema
  -> validate JSON sidecar emitted by the tracked plugin for exact node/lease evidence
  -> retain JUnit only as a human-readable aid during the run
  -> terminate exact DB sessions
  -> drop exact owned DB and role
  -> verify exact/run-prefix 0:0:0:0
```

### 4.1 Release controller

새 controller entrypoint는 다음 profile을 제공한다.

```text
uv run --locked python scripts/backend_release_matrix.py --profile settings-diagnostic
uv run --locked python scripts/backend_release_matrix.py --profile postgres
uv run --locked python scripts/backend_release_matrix.py --profile compatibility
uv run --locked python scripts/backend_release_matrix.py --profile non-slack
uv run --locked python scripts/backend_release_matrix.py --profile full
```

Task 16 Steps 4–5도 raw pytest가 아니라 각각 `postgres`와 `compatibility`
profile을 사용한다. 모든 profile이 같은 container validation, resource ownership,
environment, schema lease, evidence, cleanup 계약을 공유한다.

controller는 shell interpolation 없이 argument 배열로 Docker/pytest를 호출한다.
controller module 자체는 `backend.app.core.config`, `backend.app.db.session`,
`backend.app.db.init_db`를 import하지 않는다. Settings 또는 app database module이
필요한 작업은 아래 guarded test bootstrap child로만 실행한다.
shared `paraworks-postgres` container가 다음 계약과 정확히 일치하고
`running:healthy`일 때만 사용한다.

```text
container: paraworks-postgres
image: pgvector/pgvector:pg17
compose service: postgres
host port: 127.0.0.1:55432
server identity: paraworks:postgres
```

container가 없거나 계약이 다르면 controller는 resource를 생성·대체하지 않고
중단한다. `docker compose down`, container 제거, volume 삭제는 금지한다.

run identity는 12자리 lowercase hex이고 resource 이름은 다음 규칙을 따른다.

```text
run prefix: paraworks_c5t16_<run_id>
database:   paraworks_c5t16_<run_id>_database_test
role:       paraworks_c5t16_<run_id>_role_test
sentinel:   paraworks_c5t16_<run_id>_sentinel
lease:      paraworks_c5t16_<run_id>_schema_<scope>_<hex>
```

database와 role은 PostgreSQL 63-byte 제한 안에 있고 `_test`로 끝나야 한다.
생성 전에 exact/literal-prefix database/role count가 모두 0인지 확인한다.
pre-existing identity는 unowned이며 자동 terminate/drop하지 않는다.

controller는 OS의 secure unique-directory primitive로 non-pre-existing private temp
directory를 만들고, profile의 각 pytest child에 tracked invocation manifest의
`run_id`, `profile`, `child_id`, `invocation_hash`, unique sidecar/JUnit path를
전달한다. child id와 path는 재사용하지 않는다. controller는 기대한 child set을
시작 전에 latch하고 manifest 밖 child 또는 pre-existing artifact target을
거절한다.

### 4.2 Hermetic child environment profile

각 pytest child는 literal system allowlist에서 새 environment mapping을 만든다.
parent environment를 복사하거나 직접 수정하지 않는다. Windows에서는 실행에
필요한 `PATH`, `SystemRoot`, `WINDIR`, `TEMP`, `TMP`, `USERPROFILE`,
`APPDATA`, `LOCALAPPDATA`만 전달하고, POSIX 대응 key는 별도 literal allowlist로
관리한다. controller와 plugin은 repo의 `.env`를 읽거나 key/value를 출력하지
않는다.

tracked test-only `backend.tests.release_bootstrap`은 dotenv guard와 `probe`,
`init-sqlite` child entrypoint를 소유한다. `-p
backend.tests.release_evidence_plugin`으로 명시적으로 load된 plugin도 같은 guard
함수를 `pytest_load_initial_conftests` 시점, 즉 `backend/tests/conftest.py`가
`backend.app.db.session`을 import하기 전에 호출한다.

1. `backend.app.core.config.get_settings.cache_info().currsize == 0`인지 확인한다.
2. `Settings.model_config['env_file']`을 process-local하게 `None`으로 바꾼다.
3. effective `env_file=None`을 다시 확인하고 `get_settings.cache_clear()`한다.
4. hook이 늦었거나 guard가 적용되지 않으면 collection 전에
   `environment_refused`로 중단한다.

따라서 test가 `monkeypatch.delenv()`를 호출해도 실제 repo `.env`가 fallback으로
다시 나타나지 않는다. production Settings source나 repo `.env` 자체는 수정하지
않는다.

pytest 밖에서 Settings를 읽는 probe와 SQLite 초기화도 각각 다음 test-only child를
사용한다.

```text
python -m backend.tests.release_bootstrap probe
python -m backend.tests.release_bootstrap init-sqlite
```

bootstrap은 먼저 위 guard를 적용하고 그 뒤에만 `Settings`,
`backend.app.db.init_db`, `backend.app.db.session`을 필요한 순서로 import한다. direct
`python -m backend.app.db.init_db`와 guard 없는 Settings probe는 금지한다.

허용되는 database 값:

- `DATABASE_URL`: child 전용 임시 file-backed SQLite URL
- `PARAWORKS_TEST_POSTGRES_URL`: controller-owned PostgreSQL base URL
- `PARAWORKS_PGVECTOR_TEST_DATABASE_URL`: 동일 base resource locator
- `PARAWORKS_DEMO_MODE=true`
- `PARAWORKS_DATABASE_URL`과 `PARAWORKS_DEMO_DATABASE_URL`: child mapping에서 제외
- `PARAWORKS_ENV=local`
- `AGENT_RUNTIME_FINGERPRINT_SECRET`와 key version: child mapping에서 제외하여
  Settings의 local defaults 사용
- `AUTO_REVIEW_MODE=disabled`
- `AUTO_REVIEW_ENFORCE_PERCENTAGE=0`
- `AGENT_LLM_ENABLED=false`
- live provider/connector/OAuth credential key는 child mapping에서 제외

다음 pytest control 값은 제거하거나 exact safe value로 덮어쓴다.

- `PYTEST_ADDOPTS=`
- `PYTEST_PLUGINS=`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- `PYTEST_XDIST_WORKER`와 xdist worker-count 관련 key 제거
- paid-evaluation flag와 provider confirmation flag 제거

pytest는 위 internal plugin만 명시적으로 load한다. 다른 plugin이 필요한 경우
collection-parity RED와 독립 review를 거쳐 exact plugin manifest에 추가한다.
xdist가 감지되면 collection 전에 거절한다.

별도 precollection probe는 secret value를 출력하지 않고 다음 effective state만
확인한다.

```text
resolved_database_backend=sqlite
dotenv_source_enabled=false
demo_mode=true
auto_review_mode=disabled
agent_llm_enabled=false
live_provider_credentials_present=false
paid_provider_flags_present=false
pytest_worker_count=1
```

explicit constructor 값은 process environment 때문에 다른 Settings field로
우회되지 않아야 한다. 임시 SQLite schema는 pytest collection 전에 위 guarded
`init-sqlite` child로 준비한다. pytest child가 끝나면 SQLite file과 `-journal`,
`-wal`, `-shm` companion을 exact path로 삭제한다.

### 4.3 Schema lease contract

test-only helper는 다음 공개 계약을 제공한다.

```python
@dataclass(frozen=True)
class PostgresSchemaLease:
    base_url: str
    database_url: str
    schema_name: str

@contextmanager
def lease_postgres_schema(
    base_url: str,
    *,
    run_id: str,
    scope_name: str,
) -> Iterator[PostgresSchemaLease]: ...
```

`lease_postgres_schema`의 계약은 다음과 같다.

1. URL parse 전후에 PostgreSQL backend인지 확인한다.
2. controller는 non-secret exact database/role identity를 child에 전달한다.
3. parsed URL database/user, `current_database()`/`current_user`, controller가
   latch한 exact database/role이 각각 모두 동일한지 확인한다.
4. 일치한 database/user가 모두 `_test`로 끝나는지 확인한다.
5. non-test database/user allowlist를 거절한다.
6. base database의 public application table preflight는 session 시작 시 한 번만
   수행한다.
7. schema 이름은 controller가 latch한 run id, sanitized scope, random lowercase
   hex를 사용한 `paraworks_c5t16_<run_id>_schema_<scope>_<hex>` literal prefix로
   만들고 63 bytes 이하로 제한한다.
8. exact schema가 이미 있으면 생성·채택·삭제하지 않고 중단한다.
9. `CREATE SCHEMA` 성공 뒤에만 ownership을 latch한다.
10. leased URL은 기존 query를 보존하면서
   `options=-csearch_path=<schema>,public`을 설정한다.
11. consumer가 session, engine, LangGraph pool/saver를 닫은 뒤 context를
   종료해야 한다.
12. context 종료 시 ownership이 latch된 exact schema만 `DROP SCHEMA ...
    CASCADE`하고 같은 admin connection에서 `pg_namespace` exact-name 부재를 확인한
    뒤에만 drop event를 기록한다.
13. cleanup admin connection은 `lock_timeout=5s`, `statement_timeout=30s`를
    적용하여 leaked transaction에 무기한 block되지 않는다.
14. cleanup은 한 번만 시도하고, cleanup failure가 이전 test/setup 결과보다
    우선한다.
15. rendered exception/output에 base URL, password, driver message를 넣지 않는다.

pytest process는 serial이어야 한다. schema가 달라도 PostgreSQL advisory lock은
database 전체에 적용되므로 xdist worker는 격리되지 않는다. 향후 병렬 release가
필요하면 별도 설계로 worker별 physical database를 사용한다.

### 4.4 Lease 적용 범위

현재 PostgreSQL resource locator를 사용하는 다음 모듈은 public schema 대신
leased URL을 사용한다.

- `test_agent_runtime_postgres_checkpoint.py`: module lease
- `test_auto_review_migration.py`: module lease
- `test_auto_review_provenance.py`: module lease
- `test_review_transition_postgres.py`: function lease
- `test_review_v21_extraction_postgres.py`: function lease 유지
- `test_review_v2_postgres.py`: function lease
- `test_pgvector_integration.py`: function 또는 module lease
- Task 16에서 추가될 `test_auto_review_postgres.py`: function/module 특성에 맞는
  lease

scope 선택은 freshness 의미를 보존한다. restart/concurrency를 검증하는 하나의
test 내부 connection은 동일 schema를 공유하지만 다른 test와는 공유하지 않는다.
Alembic, SQLAlchemy sessionmaker, checkpoint pool, pgvector adapter는 전역
`db.session`을 사용하지 않고 leased URL/engine을 명시적으로 받는다.
durable C.5 key가 필요한 PostgreSQL fixture만 자기 scope 안에서 strong test-only
fingerprint secret/key version을 명시적으로 설정하고 종료 시 복원한다. controller는
그 값을 전체 pytest child에 주입하지 않으며 sidecar/report에도 기록하지 않는다.

현재 `backend/migrations/env.py`는 `Settings.resolved_database_url()`을
`alembic.ini`의 `sqlalchemy.url`보다 우선한다. 따라서 Alembic 실행 fixture는
다음 protocol을 하나의 context로 고정한다.

1. 기존 `PARAWORKS_DEMO_MODE`, `PARAWORKS_DATABASE_URL`, `DATABASE_URL`을
   snapshot한다.
2. `PARAWORKS_DEMO_MODE=false`, `PARAWORKS_DATABASE_URL=<leased URL>`,
   `DATABASE_URL=<leased URL>`을 임시 설정한다.
3. `get_settings.cache_clear()` 후 Alembic을 실행한다.
4. SQLAlchemy connection의 `current_schema()`, Alembic version table schema,
   LangGraph checkpoint connection의 `current_schema()`가 exact leased schema와
   같은지 확인한다.
5. 모든 pool/engine을 닫고 environment를 복원한 뒤 다시
   `get_settings.cache_clear()`한다.

pytest가 serial이라는 전제 때문에 이 process-global fixture patch가 허용된다.
다른 test가 실행되는 동안 이 context를 열어 두지 않는다.

### 4.5 Review V2 test fixture correction

`test_review_v2_postgres.py`의 deterministic drafting fixture는 production
evidence trigger를 만족하는 최소 shape를 만든다.

- 먼저 canonical `AgentWorkflowEvidenceRef`를 만든다.
- `ReviewItem.agent_run_id`에 실제 same-workflow AgentRun id를 저장한다.
- `candidate_contract_version='c5-v1'`을 저장한다.
- 같은 workflow와 evidence ordinal을 가리키는 `ReviewItemEvidenceRef`를 만든다.
- JSON payload 안의 legacy id만으로 provenance를 흉내 내지 않는다.
- cleanup은 evidence child를 parent보다 먼저 처리하거나 owned schema drop에
  위임한다.

기존 restart, privacy, repair, exact-batch, concurrent-resume assertion은 바꾸지
않는다. pre-fix `6 failed, 3 passed`를 RED로 보존하고 post-fix `9 passed, 0
skipped`를 요구한다.

## 5. 실행 흐름

### 5.1 Controller preflight

1. 모든 test/helper/controller behavior slice가 review된 intermediate commit으로
   끝났는지 확인하고 clean tracked tree와 exact verification commit을 기록한다.
2. Docker/container/server identity를 read-only로 검증한다.
3. generated run prefix의 exact/literal-prefix count `0:0:0:0`을 확인한다.
4. role을 생성하고 성공 후 ownership을 latch한다.
5. database를 생성하고 성공 후 ownership을 latch한다.
6. pgvector extension을 생성한다.
7. public application table이 비어 있는지 한 번 확인한다.
8. exact non-lease sentinel schema를 생성하고 ownership을 latch한다.

### 5.2 Diagnostic gates

구현 전후에 controller contamination을 분리하기 위해 다음 여섯 Settings node를
sanitized child environment에서 실행한다.

```text
test_enabled_checkpoint_mode_rejects_unsupported_backend_without_secret
test_production_rejects_the_local_default_fingerprint_secret
test_non_disabled_mode_rejects_local_default_fingerprint_secret
test_disabled_sqlite_smoke_may_use_process_local_placeholder_without_durable_ready_state
test_process_local_sqlite_smoke_is_limited_to_disabled_in_memory_url[...True]
test_file_backed_sqlite_placeholder_refuses_every_c5_bound_durable_write
```

Expected: `6 passed`. assertion을 수정하지 않는다.

Review V2 fixture는 dedicated clean leased schema에서 `-vv --tb=long`으로 먼저
단독 실행하고 bounded RED artifact를 만든다.

```text
pre-fix: exact six node failure phase/cause를 sidecar에 기록
hypothesis confirmed: post-C.5 evidence guard에서 6 failed, 3 passed
post-fix when confirmed: 9 passed, 0 skipped
```

가설과 다른 failure이면 fixture 변경을 중단하고 owning contract를 다시 계획한다.

### 5.3 Non-Slack mode

canonical collection에서 사용자 승인된 Slack node 열 개만 deselect한다. 다른
node exclusion, ignore, skip, xfail은 허용하지 않는다. authoritative evidence는
tracked internal pytest plugin이 child별 unique target에 쓰는 privacy-safe JSON
sidecar다. sidecar envelope에는 schema version, `run_id`, `profile`, `child_id`,
`invocation_hash`, exact `report.nodeid`, phase, outcome, `wasxfail`, collection
index와 schema lease의 hashed lease id/create/drop 상태가 들어간다. Review V2의
clean RED 원인 확인에는 exact node/call phase와 exact PostgreSQL provenance-guard
문구를 메모리에서만 매핑한 allowlisted reason code 하나를 사용할 수 있다.
`longrepr`/exception text 자체는 sidecar나 report에 기록하지 않는다.
traceback, captured output, URL, schema/database/role name, secret은 넣지 않는다.
JUnit XML은 사람이 읽는 보조 artifact일 뿐 coverage authority가 아니다.

plugin은 event를 같은 private directory의 partial file에 기록하고
`pytest_sessionfinish`에서 pytest exit status, collection hash, event hash,
`complete=true` completion marker를 넣은 뒤 fsync하고 non-pre-existing final target으로
atomic rename한다. controller는 partial/missing/duplicate sidecar, identity/hash mismatch,
completion marker 부재, native child exit와 pytest exit status 불일치를 모두
`evidence_refused`로 처리한다. 따라서 이전 실행이나 중간 JSON을 채택할 수 없다.

exact node id도 privacy 검사를 통과해야 한다. 구현 전 collection RED에서 URL scheme,
DSN userinfo, credential/secret-like parameter가 들어간 node id를 찾고 해당
`@pytest.mark.parametrize`에 stable non-sensitive explicit `ids=`를 추가한다. plugin은
collection과 report node id에 URL/DSN/secret pattern이 남아 있으면 값을 sidecar에
쓰기 전에 `evidence_refused`로 중단한다. safe-id 변경 전후의 `(test function,
case count)` parity를 controller test로 보존하며 Slack baseline 변경은 별도 human
decision으로 남긴다.

controller는 sidecar schema version과 hash를 검증한 뒤 다음을 확인한다.

- selected node set = canonical node set - exact Slack baseline
- failure/error = 0
- PostgreSQL release-critical skip = 0
- skip/xfail = 0
- 모든 leased schema가 종료 시 사라짐

profile에 pytest child가 둘 이상이면 각 child가 고유 sidecar를 만들고, controller는
latched child set과 sidecar set의 exact equality, child node set의 disjointness와
expected profile collection에 대한 exact union을 검증한다. 특히 `compatibility`
profile의 여러 child 결과를 단순 pass-count 합으로 대체하지 않는다.

### 5.4 Full mode

전체 canonical collection을 실행한다. controller는 pytest exit 1을 자동 성공으로
간주하지 않는다. authoritative JSON sidecar에서 다음 조건을 모두 확인할 때만
deferred-baseline match로 분류한다. JUnit은 판정에 사용하지 않는다.

- failure node set = exact Slack baseline ten
- error node set = empty
- skip/xfail = empty
- C.5/non-Slack node failure = empty
- collection 누락/중복 = empty

Slack baseline 자체의 변경은 human decision이다. 새 failure를 baseline에 추가하지
않는다.

## 6. 실패·정리 우선순위

### 6.1 Failure classes

controller 결과는 bounded code와 count만 출력한다.

```text
preflight_refused
environment_refused
collection_mismatch
evidence_refused
verification_failed
cleanup_failed
baseline_mismatch
passed
```

test failure node id와 count는 출력할 수 있지만 assertion payload, source content,
DSN, password, provider response는 출력하지 않는다.

### 6.2 Cleanup precedence

schema cleanup은 각 lease가 소유하고 database/role cleanup은 outer controller가
소유한다. outer cleanup 순서는 다음과 같다.

1. 정상 child exit를 확인하거나, profile timeout 시 controller가 생성한 exact pytest
   process group을 먼저 종료하고 종료 여부를 확인
2. bounded inspection connection을 exact owned database에 열고 backend pid를 latch
3. maintenance admin connection으로 exact database의 새 connection을 금지하고
   inspection pid를 제외한 session을 terminate하여 quiesce
4. inspection connection에서 `pg_namespace`의 controller-latched lease-schema literal
   prefix count가 0이고 non-lease sentinel schema가 아직 존재하는지 확인; sidecar
   event만 catalog authority로 사용하지 않음
5. inspection connection을 닫고 exact owned database drop
6. exact owned role drop
7. exact/run-prefix database/role catalog postflight
8. temporary SQLite/JUnit/collection/sidecar artifact exact-path deletion

profile timeout은 `settings-diagnostic=120s`, 나머지 profile은 `1200s`다. inspection,
maintenance, terminate, catalog query, database/role drop을 포함한 모든 outer admin
connection/statement에도 각각 bounded connect/lock/statement timeout을 적용한다.
timeout 후 exact child termination이 확인되지 않으면 database cleanup은 계속
시도하되 최종 결과는 `cleanup_failed`다. 각 cleanup은 이전 cleanup failure가
있어도 계속 시도한다. cleanup failure는
verification success/failure보다 우선한다. residual resource가 있으면 controller는
자동 재채택·재삭제하지 않고 exact bounded identity만 보고한 뒤 human decision을
요구한다. child 종료가 확인되지 않으면 catalog 결과를 clean evidence로 인정하지
않는다. schema prefix count가 0이 아니거나 sentinel이 사라졌어도 outer exact
database/role cleanup은 계속하지만 최종 결과는 `cleanup_failed`다. 여기서 sentinel
생존은 lease cleanup까지의 비간섭 증거이며, controller-owned database를 최종 drop할
때 sentinel도 함께 삭제되는 것은 의도된 outer cleanup이다.

## 7. Coverage와 증거 계약

tracked manifest는 다음을 보관한다.

- exact deferred Slack failure node ten
- PostgreSQL release-critical module/node classification
- exact profile-to-child invocation manifest and expected node partition
- controller output schema version
- pytest sidecar schema version
- explicitly loaded pytest plugin list

baseline은 test failure를 숨기는 용도가 아니다. manifest 변경은 독립 review와
human approval을 요구한다.

각 release report는 다음만 보존한다.

- commit SHA
- mode
- collected/selected/passed/failed/error/skipped/deselected counts
- exact unexpected node ids
- allowlisted failure-reason code별 aggregate count
- schema lease created/dropped counts
- expected/observed child-sidecar identity and completion counts
- pre-drop lease-schema prefix count and non-lease sentinel state
- database/role ownership 결과
- exact/run-prefix cleanup `0:0:0:0`
- no-live-provider 확인
- lock/Ruff/compile/diff 결과

raw JUnit, collection, sidecar artifact는 system temp에서 hash를 계산한 뒤
삭제한다. aggregate hash만 report에 남긴다. lease created/dropped count는 sidecar의
hashed lease id state machine에서 계산하며, duplicate create/drop, missing drop,
unknown lease id는 controller failure다. 이 event 계산은 pre-drop catalog/sentinel
확인을 대체하지 않는다.

## 8. 보안과 권한

- controller는 DB password를 생성하되 stdout/stderr, report, Git에 기록하지 않는다.
- subprocess는 `shell=True`를 사용하지 않는다.
- Docker/psql argument와 SQL identifier는 allowlisted/generated 값만 사용한다.
- source URL/snippet, prompt, model output, OAuth/provider credential을 release
  artifact에 쓰지 않는다.
- live provider credential과 paid-evaluation flag는 child environment에서 제거한다.
- 테스트는 fake/deterministic provider만 사용한다.
- shared container, pre-existing database/role/schema, volume을 삭제하지 않는다.
- restricted source 또는 Review trust boundary를 변경하지 않는다.

## 9. 검토한 대안

### 9.1 모듈마다 physical database

격리와 future xdist에는 가장 강하지만 database 생성/migration/extension 비용,
connection pressure, privileged URL mapping이 커진다. 현재 serial release에는
과도하다. xdist가 실제 요구될 때 별도 설계한다.

### 9.2 Controller-partitioned pytest invocation

변경량은 작지만 raw `pytest backend/tests`가 계속 순서 의존적으로 실패하며,
coverage union을 별도로 증명해야 한다. 임시 우회로 채택하지 않는다.

### 9.3 Fresh-table guard 삭제만 수행

Settings contamination, shared rows/checkpoints, future module ordering을 해결하지
못한다. fail-closed guard를 단순 제거하지 않는다.

## 10. 예상 파일 경계

작성된 상세 구현 계획은 다음 범위 안에서 작업을 나눈다.

- Create: `backend/tests/postgres_isolation.py`
- Create: `backend/tests/test_postgres_isolation.py`
- Create: `backend/tests/release_contracts.py`
- Create: `backend/tests/test_release_contracts.py`
- Create: `backend/tests/release_bootstrap.py`
- Create: `backend/tests/test_release_bootstrap.py`
- Create: `backend/tests/release_evidence_plugin.py`
- Create: `backend/tests/test_release_evidence_plugin.py`
- Create: `scripts/backend_release_matrix.py`
- Create: `backend/tests/test_backend_release_matrix.py`
- Create: `backend/tests/fixtures/deferred_backend_baseline_v1.json`
- Modify: safe explicit parameter id가 필요한 test modules, collection RED로 범위 확정
- Modify: PostgreSQL fixture를 가진 테스트 모듈
- Modify: `docs/superpowers/plans/2026-08-28-auto-review-trust-promotion.md`
- Modify after observed implementation: `plan.md`, portfolio log, session handoff
- Create after implementation: `docs/superpowers/runbooks/backend-release-matrix.md`

production `backend/app`와 Alembic migration은 기본 범위에 없다. isolated valid
fixture에서도 behavior assertion이 실패할 때만 별도 TDD task와 human review를
거쳐 product code 범위를 연다.

## 11. 수용 기준

- [ ] 여섯 Settings diagnostic node가 assertion 수정 없이 `6 passed`다.
- [ ] schema lease가 invalid/non-`_test` URL에서 mutation 없이 거절한다.
- [ ] pre-existing schema를 채택하거나 삭제하지 않는다.
- [ ] 두 lease가 서로의 marker/Alembic/checkpoint row를 볼 수 없다.
- [ ] setup/test/cleanup failure 모두 exact owned schema만 한 번 정리한다.
- [ ] controller-owned non-lease sentinel은 lease의 모든 adverse cleanup에서 pre-drop
      checkpoint까지 살아남는다.
- [ ] 모든 PostgreSQL 모듈이 public 대신 leased URL을 사용한다.
- [ ] collection-bound `db.session.engine`/`SessionLocal`을 PG fixture가 사용하지
      않는다.
- [ ] adverse module-order permutation에서 setup error가 0이다.
- [ ] Review V2 PostgreSQL fixture가 실제 C.5 provenance를 만들고 `9 passed, 0
      skipped`다. 단, targeted RED가 provenance 가설을 확인한 뒤에만 수정한다.
- [ ] Task 16 PostgreSQL gate가 zero-skip다.
- [ ] non-Slack gate가 exact Slack ten 외 추가 deselection 없이 통과한다.
- [ ] full gate의 failure set이 exact Slack ten이고 error가 0이다.
- [ ] xdist가 명시적으로 거절된다.
- [ ] `.env`, `PYTEST_ADDOPTS`, `PYTEST_PLUGINS`, credential을 주입한 adverse
      environment에서도 effective hermetic probe가 같거나 collection 전에
      fail-closed한다.
- [ ] test가 credential environment를 `delenv()`한 뒤에도 dotenv source가 다시
      나타나지 않는다.
- [ ] adverse `.env` 아래에서도 guarded probe와 SQLite init child가 safe state를
      유지하고 direct unguarded app init path는 controller에서 호출되지 않는다.
- [ ] 모든 collected/reported node id가 URL/DSN/secret privacy validator를 통과하고
      safe-id 전후 test function별 case count가 같다.
- [ ] exact node/phase/outcome과 hashed lease lifecycle sidecar가 collection과
      cleanup을 완전히 설명한다.
- [ ] sidecar의 run/profile/child/invocation identity, completion marker, native-exit
      consistency가 stale/partial artifact 채택을 거절한다.
- [ ] multi-child profile의 sidecar set과 node partition union/disjointness가 exact다.
- [ ] outer controller의 pre-drop `pg_namespace` prefix count가 sidecar와 독립적으로
      schema cleanup 0을 증명한다.
- [ ] Alembic, SQLAlchemy, LangGraph connection의 `current_schema()`가 모두 leased
      schema와 같다.
- [ ] timeout/leaked transaction 검증이 bounded child termination과 outer cleanup을
      거쳐 끝난다.
- [ ] live provider call/credential inheritance가 0이다.
- [ ] schema postflight와 outer database/role postflight가 모두 clean이며 최종
      `0:0:0:0`이다.
- [ ] lock, Ruff, compile, diff, executable secret/privacy scan이 통과한다.

## 12. 실행 순서와 승인 경계

제품 실행 순서는 그대로다.

```text
C.5 Tasks 6–15
  -> Task 16 entry: whole-suite PostgreSQL isolation implementation
  -> Task 16 release proof
  -> Deliverable D
  -> Deliverable E
  -> Slack reconstruction/regression last
```

이 spec은 사용자 승인되었고 별도 implementation plan이
`docs/superpowers/plans/2026-08-29-whole-suite-postgresql-isolation.md`에 작성되었다.
그 implementation plan 승인 전에는 test/helper/controller code를 변경하지 않는다.
paid provider gate, Slack baseline 변경, output schema, permission, token/cost,
Review trust, promotion/revoke/duplicate 정책 변경은 각각 별도 human decision이
필요하다.

기존 C.5 Task 16 plan의 raw pytest Steps 4–7과 마지막 단일 commit 지시는 이 경계를
반영하지 못했다. 작성된 implementation plan과 같은 planning change에서 해당 steps를
controller profile 호출로 바꾸고 behavior slice별 reviewed intermediate commit 뒤
clean verification commit을 요구하도록 좁게 amend했다. 그 amendment와 implementation
plan이 별도 승인되기 전에는 Task 16 release 명령을 실행하지 않는다.

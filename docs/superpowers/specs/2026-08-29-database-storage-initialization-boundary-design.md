# Database Storage Initialization Boundary Design

검토 버전: 2
작성일: 2026-08-29
상태: 독립 최종 검토 PASS · 사용자 최종 승인 완료 · 구현 계획 독립 검토 PASS — 실제 구현 승인 대기

## 1. 결정 요약

Deliverable C.5 Task 5의 key-admin CLI는 설정 오류와 저장소 초기화 오류를
서로 다른 bounded 결과로 보고해야 한다. 현재 구현은
`backend.app.db.session` 모듈 import 중 설정 로딩, DBAPI import, SQLAlchemy engine
생성이 함께 실행되므로 예외 클래스만으로 오류의 출처를 정확히 구분할 수 없다.

ParaWorks는 DB 계층에 설정과 분리된 **typed storage initializer**를 추가한다.
initializer는 선택된 database URL을 입력받아 구문/방언 설정과 실제
DBAPI/driver/engine 가용성을 단계별로 구분한 뒤 SQLAlchemy engine과 session factory를
함께 생성한다. 잘못된 URL·방언 설정과 저장소 가용성 실패는 서로 다른 내용 없는
전용 예외로 변환한다. key-admin CLI는 전자만 `configuration_refused`/exit 2,
후자만 `storage_unavailable`/exit 3으로 분류한다.

CLI가 소유한 runtime은 성공과 모든 실패 경로에서 정확히 한 번 dispose한 뒤 결과를
한 번만 출력한다. 기존 애플리케이션의 전역 runtime은 현재와 같이 process lifetime
동안 유지한다.

기존 애플리케이션의 `engine`, `SessionLocal`, `get_db` 공개 사용법은 유지한다.
Task 5 외 기능의 lazy-session 전환이나 전역 DB lifecycle 재설계는 하지 않는다.

## 2. 검토한 접근

### A. DB-owned typed initializer — 선택

- 새 DB 모듈이 URL → engine/session factory 생성만 소유한다.
- 설정 로딩이나 CLI parsing은 모듈 밖에서 완료한다.
- URL/방언 설정 오류와 저장소 초기화 실패는 서로 다른 typed exception으로 전달한다.
- runtime ownership과 idempotent disposal을 한 계약으로 제공한다.
- 기존 `db.session`은 새 initializer를 사용하는 호환 adapter가 된다.

장점은 오류 출처가 명확하고 CLI와 애플리케이션이 같은 DB 생성 경로를 공유한다는
점이다. 변경 범위도 DB 초기화와 Task 5 CLI에 한정된다.

### B. CLI 안에서 `create_engine()` 중복 — 제외

공용 DB 모듈을 건드리지 않지만, CLI와 애플리케이션의 pool/connect-args/sessionmaker
설정이 갈라진다. 향후 한쪽만 수정될 위험이 있어 제외한다.

### C. 애플리케이션 전체 lazy DB runtime — 제외

import-time engine 생성을 완전히 없앨 수 있으나 FastAPI, Celery, migration, 테스트의
lifecycle까지 바꾸는 별도 전달 단위다. Task 5 수용 결함을 해결하기에는 범위가 너무
넓다.

## 3. 구성 요소와 공개 계약

### 3.1 전용 초기화 모듈

`backend/app/db/initialization.py`를 추가한다.

공개 계약은 다음 세 개다.

- `DatabaseRuntime`
  - `engine`: 생성된 SQLAlchemy `Engine`; `repr`에서 제외
  - `session_factory`: 해당 engine에 bind된 `sessionmaker[Session]`; `repr`에서 제외
  - `dispose()`: engine/pool을 정확히 한 번 정리하는 idempotent operation
- `initialize_database_runtime(database_url: str) -> DatabaseRuntime`
- `DatabaseConfigurationError`와 `DatabaseInitializationError`

`DatabaseConfigurationError`는 URL 구문, 존재하지 않는 dialect/plugin, 지원되지 않는
engine 설정처럼 operator가 배포 설정을 고쳐야 하는 실패다.
`DatabaseInitializationError`는 유효한 dialect에서 DBAPI package 부재, native module/DLL
load 실패 또는 engine/pool 초기화 불가처럼 저장소 runtime을 사용할 수 없는 실패다.
두 예외는 각각 class-level 고정 code
`database_configuration_invalid`와 `database_initialization_failed`만 가진다. initializer가
입력받은 database URL·username·password·driver/module 이름과 initializer가 잡은 원본
exception은 새 예외의 `args`, `repr`, `__dict__`, `__cause__`, `__context__`에 보존하지
않는다. 호출자 측 active exception이 붙을 수 있는 범위는 아래에서 별도로 정의한다.

initializer가 소유하는 작업은 다음으로 제한한다.

1. 전달받은 URL의 SQLAlchemy 구문과 dialect/plugin 설정 판정
2. DB dialect/DBAPI 로딩과 engine 생성
3. engine에 bind된 session factory 생성

`Settings`, 환경변수, CLI 인자, 애플리케이션 상태는 읽지 않는다. URL 선택은 호출자가
완료하지만 SQLAlchemy-valid 여부는 initializer가 판정한다. URL parsing과
`NoSuchModuleError`/argument 오류는 `DatabaseConfigurationError`로 변환한다. engine
생성 단계의 나머지 오류는 아래 first-match 순서로 판정한다.

1. DBAPI import/load 오류(`ModuleNotFoundError`, `ImportError`, native loader `OSError`)
2. 모든 `DBAPIError` subclass 중 `connection_invalidated=True`
3. `OperationalError`, `InterfaceError`, pool `sqlalchemy.exc.TimeoutError`,
   `DisconnectionError`
4. `InvalidRequestError`, availability로 증명되지 않은 `StatementError`/`DBAPIError`와
   나머지 `SQLAlchemyError`

1~3만 `DatabaseInitializationError`로 변환한다. 4와 임의의 programmer exception은
변환하지 않고 상위 `operation_failed` 경계로 전달한다. 따라서
`ProgrammingError(connection_invalidated=True)`는 2에서 availability로 판정되지만,
같은 오류의 `connection_invalidated=False`는 4에서 operation failure로 판정된다.

원본 예외는 active `except` block 안에서 새 예외를 raise하는 방식으로 chain하지
않는다. 실패 종류만 local sentinel로 남기고 `except` scope를 벗어난 뒤 내용 없는 새
typed exception을 raise한다. 이 방식은 initializer가 잡은 원본 storage/configuration
예외와 그 민감정보가 새 예외의 `__cause__`/`__context__`에 도달하지 않게 한다. 단,
호출자가 이미 다른 `except`를 처리하는 중 initializer를 호출하면 Python이 그 호출자
예외를 새 `__context__`로 설정할 수 있으므로, 임의 호출자 context까지 항상 `None`이라는
계약은 하지 않는다. CLI는 active exception 밖의 정상 제어 흐름에서 initializer를
호출한다. initializer는 임의의 `Exception`을 흡수하거나 storage 장애로 재분류하지
않는다. 다만 partial engine 정리를 보장하기 위해 session factory 생성 및 cleanup의
임의 예외를 lifecycle sentinel로만 잠시 포착할 수 있으며, 아래 우선순위에 따라 active
`except` 밖에서 원형 그대로 다시 raise한다.

engine 생성 후 session factory 구성에 실패하면 그 생성 오류를 local sentinel에 담고
active `except`를 벗어난 뒤 생성된 engine을 즉시 한 번 dispose한다. partial cleanup이
성공하면 원래 비저장소 생성 오류를 active `except` 밖에서 그대로 상위 sanitizer로
보낸다. partial cleanup이 실패하면 cleanup 오류가 원래 생성 오류보다 우선하며
재시도하지 않는다. 이때 cleanup 오류도 active `except` 밖에서 판정·raise하여 원래
session factory 오류를 자동 chain하지 않는다. availability cleanup 오류는 내용 없는
`DatabaseInitializationError`로 변환하고, programmer cleanup 오류는 원형 그대로 상위
`operation_failed` 경계로 보낸다.
`DatabaseRuntime.dispose()`와 partial cleanup은 위 first-match의 1~3을 같은 순서로
적용해 engine disposal의 availability 실패를 원본과 연결되지 않은
`DatabaseInitializationError`로 변환하며, 4와 임의의 programmer exception을 storage로
바꾸지 않는다. idempotence flag는 disposal attempt 전에 latch하여 한 번 실패한 engine을
두 번째 호출에서 다시 dispose하지 않는다.

initializer는 `create_engine(..., pool_pre_ping=True)`와
`sessionmaker(bind=engine, autoflush=False, autocommit=False,
expire_on_commit=True)`를 사용한다. 초기화 과정에서 `connect()`, checkout/ping, schema
inspection 또는 SQL을 실행하지 않는다. `pool_pre_ping`은 pool checkout 단계에서 필요할
때 적용되며 초기화 시에는 실행되지 않는다.

### 3.2 기존 애플리케이션 호환 adapter

`backend/app/db/session.py`는 기존처럼 `get_settings()`로 URL을 결정하지만 engine과
session factory 생성은 새 initializer에 위임한다.

다음 기존 import는 유지한다.

- `from backend.app.db.session import engine`
- `from backend.app.db.session import SessionLocal`
- `from backend.app.db.session import get_db`

따라서 API route, connector, agent runtime, migration 보조 코드의 변경은 요구하지
않는다. 애플리케이션 import 중 저장소 초기화가 실패하면 typed exception이 전파되지만,
CLI 이외의 기존 startup 처리 방식은 이번 범위에서 바꾸지 않는다.

adapter는 하나의 private process-global `DatabaseRuntime`을 보유하고, 그 runtime의
`engine`과 `session_factory`를 기존 이름으로 export한다. `SessionLocal.kw['bind'] is
engine`, `get_db()`의 request session close semantics, import-time lazy connection 특성을
보존한다. 전역 runtime은 이번 전달 단위에서 자동 dispose하지 않는다.

`Settings.resolved_database_url()`의 기존 우선순위도 그대로다. demo mode이면서 demo URL이
있으면 그것을, non-demo이면서 `paraworks_database_url`이 있으면 그것을, 아니면
`database_url`을 initializer에 정확히 전달한다.

### 3.3 key-admin CLI

`backend/app/admin/auto_review_keys.py`는 `SessionLocal`을 import해 초기화하지 않는다.
이미 생성한 `Settings`에서 resolved URL을 얻은 뒤 DB initializer를 직접 호출하고,
반환된 session factory를 admin service에 주입한다.

`rotate` command만은 storage 초기화 전에 `_EnvironmentKeyRingSource.load()`를 정확히 한
번 호출해 환경 key ring을 읽고 `FingerprintKeyRing` 검증까지 끝낸다. 누락된 key ring은
기존 allowlisted `key_ring_unavailable`/exit 2를 유지하고, 빈 version·32-byte 미만 또는
동일한 secret의 `ValueError`는 `configuration_refused`/exit 2로 분류한다. 검증된 key
ring은 private fixed source로 감싸 service에 주입하며 command 단계에서 환경을 다시 읽지
않는다. 다른 command는 key ring을 읽지 않는다. 이 사전 검증이 실패하면 DB initializer를
호출하지 않으므로 dispose할 runtime도 없다.

initializer 모듈 import는 `main()`의 bounded configuration phase 안에서 수행하지만
storage invocation 밖에 둔다. 따라서 initializer 모듈 자체의 unrelated import failure는
storage로 분류되지 않는다. 오직 initializer가 반환한 typed exception만 URL/config 또는
storage initialization 의미를 가진다.

오류 분류는 단계별로 고정한다.

- parser, Settings, initializer 모듈 import, rotate key-ring 값 검증,
  `DatabaseConfigurationError`:
  `{"ok":false,"code":"configuration_refused"}`, exit 2
- allowlisted `AutoReviewKeyAdminError`/`AutoReviewKeyBootstrapError`:
  해당 bounded code, exit 2
- `DatabaseInitializationError`, 또는 command 실행 중 아래 순서에서 처음 일치하는
  availability 오류:
  1. `DBAPIError(connection_invalidated=True)` — 구체 subclass보다 우선
  2. `OperationalError`, `InterfaceError`, pool `sqlalchemy.exc.TimeoutError`,
     `DisconnectionError`
  `{"ok":false,"code":"storage_unavailable"}`, exit 3
- `IntegrityError`, `ProgrammingError`, `DataError`, `InvalidRequestError`, availability로
  증명되지 않은 `StatementError`/`SQLAlchemyError`/`DBAPIError`, service
  construction/dispatch의 알 수 없는 failure:
  `{"ok":false,"code":"operation_failed"}`, exit 3
- 성공:
  기존 command별 aggregate JSON과 readiness 기반 exit code

모든 실패 출력은 stdout 한 줄의 두 필드 JSON만 사용한다. stderr, argparse usage,
traceback, local path, URL, driver name, 사용자 입력값, secret은 출력하지 않는다.
generic sanitizer는 privacy envelope일 뿐 semantic storage classifier가 아니다.

CLI는 command 결과를 메모리에 만든 뒤 runtime을 dispose하고, cleanup이 성공한 후에만
success/readiness JSON을 출력한다. command 또는 cleanup이 실패해도 출력은 정확히 한
줄이다. cleanup이 성공하면 원래 command 결과를 사용한다. cleanup이 실패하면 그 실패가
원래 command 결과보다 우선한다. cleanup의 `DatabaseInitializationError`는
`storage_unavailable`/exit 3, 임의의 programmer exception은 `operation_failed`/exit 3으로
분류하며 이전 success/error payload나 두 번째 오류 JSON을 출력하지 않는다.

## 4. 데이터 흐름

```text
argv
  -> bounded parser
  -> Settings validation
  -> rotate only: load and validate key ring exactly once
  -> resolve database URL
  -> initialize_database_runtime(url)
       -> URL/dialect configuration
       -> DBAPI/engine/sessionmaker only; no connection
       -> DatabaseRuntime OR typed configuration/initialization error
  -> AutoReviewKeyAdminService(session_factory)
  -> status/bootstrap/rebuild/rotate
  -> DatabaseRuntime.dispose() exactly once
  -> allowlisted JSON + exact exit code
```

비저장소 코드가 `ModuleNotFoundError`를 발생시키더라도 initializer 호출 밖이면 storage
오류로 분류되지 않는다. 반대로 engine 생성이 소유한 native DBAPI
`ModuleNotFoundError`/`ImportError`/`OSError`는 DB 계층이 소유한 실행 단계에서
`DatabaseInitializationError`로 변환된다. malformed URL과 존재하지 않는 dialect는
configuration 오류이며 storage unavailable이 아니다.

## 5. 테스트 전략

TDD 순서는 다음과 같다.

1. 새 initializer 단위 RED
   - 정상 URL에서 engine/session factory 생성과 exact options/bind identity
   - malformed URL과 존재하지 않는 dialect는 configuration typed error
   - 없는 DBAPI package
   - native DBAPI `ImportError`
   - native loader `OSError`
   - 설치 환경에 의존하지 않는 temporary import hook/test dialect로 위 세 경계 재현
   - programmer `TypeError`와 initializer 모듈 import 단계의 unrelated module failure는
     storage typed error가 아님
   - initializer도 first-match를 적용해 DBAPI subclass별
     `connection_invalidated=True/False` matrix를 정확히 구분
   - `InvalidRequestError`/availability가 아닌 `StatementError`는 storage typed error가 아님
   - typed exception의 `args`, `repr`, `__dict__`, `__cause__`와 normal-control-flow
     `__context__`, rendered traceback에 initializer 입력 URL/driver와 initializer가 잡은
     원본 message가 없음
   - caller의 active exception이 context가 될 수 있어도 initializer가 잡은 원본
     storage/configuration exception과 민감정보는 cause/context에 없음
   - initializer import와 runtime 생성은 connect/ping/SQL을 실행하지 않음
   - engine 생성 후 sessionmaker 실패 시 active `except` 밖에서 partial engine dispose
     exactly once
   - partial cleanup 성공은 원래 생성 오류, availability 실패는 typed storage 오류,
     programmer 실패는 원형 operation failure이며 cleanup 실패가 원래 오류보다 우선
   - partial cleanup availability 오류의 `__cause__`, `__context__`, rendered traceback에
     원래 sessionmaker 오류와 민감 sentinel이 없음
2. CLI subprocess RED
   - 위 세 storage 초기화 실패가 정확히 exit 3
   - malformed URL/dialect, parser/Settings/비저장소 `ModuleNotFoundError`는 exit 2
   - command availability SQLAlchemy 오류만 storage exit 3
   - `DBAPIError.connection_invalidated=True`를 먼저 적용한 뒤 integrity/programming/
     data/session/other DBAPI의 invalidated true/false matrix 검증
   - availability가 아닌 integrity/programming/data/session misuse는
     `operation_failed` exit 3
   - 실제 `python -m` subprocess에는 temporary directory의 test-only `sitecustomize.py`를
     `PYTHONPATH` 선두로 전달해 fake dialect/import hook을 등록; 로컬 DBAPI 설치 여부와
     무관하게 ImportError/OSError를 재현하고 production package에는 test hook을 추가하지 않음
   - rotate의 missing key ring은 `key_ring_unavailable`/2, blank/short/equal key material은
     `configuration_refused`/2이며 모두 DB initializer call count 0
   - stdout exact JSON, stderr empty, 민감 sentinel 비노출
   - success, readiness failure, admin refusal, service-construction failure, command failure,
     cleanup availability failure, cleanup programmer failure 모두 runtime disposal attempt
     exactly once와 single JSON emission; dispose 재호출은 engine을 다시 건드리지 않음
3. import/compatibility RED
   - 기존 `engine`, `SessionLocal`, `get_db` symbol과 exact SQLAlchemy option 계약
   - `db.initialization -> db.session`, `auto_review_keys` direct/`python -m`,
     `backend.app.main`, `db.init_db`, Celery sync/reindex, agent bootstrap/retention,
     FastAPI `get_db` override의 direct/application-first import matrix
   - demo/non-demo/default resolved URL 우선순위와 initializer 전달값
   - 새 initializer가 Settings/environment를 읽지 않음
4. 기존 Task 5 전체 회귀
   - migration/bootstrap/lock gate
   - Task 5 provenance/transition/CLI union
   - Task 4 actor/auth/RBAC/audit
   - 모든 PostgreSQL 테스트 zero-skip

최종 검증은 새 database와 role 이름이 모두 `_test`로 끝나는 일회성 PostgreSQL +
pgvector 환경, `127.0.0.1:55432`에서 수행한다. 정확한 DB/role 삭제와 catalog `0:0`을
증명한다.

## 6. 파일 범위

허용되는 제품 코드 변경:

- 새 `backend/app/db/initialization.py`
- `backend/app/db/session.py`
- `backend/app/admin/auto_review_keys.py`

허용되는 테스트 변경:

- 새 `backend/tests/test_database_initialization.py`
- `backend/tests/test_auto_review_provenance.py`
- 기존 session/import 호환 테스트가 더 적절하면 해당 파일의 최소 수정

문서 변경:

- 이 설계 문서
- C.5 Task 5 report/progress ledger
- `docs/portfolio-log.md`
- `docs/superpowers/runbooks/session-handoff.md`

Task 6, Slack, connector, RAG serving, application-wide lazy DB lifecycle은 제외한다.

## 7. 수용 기준

- DB 초기화와 설정/모듈 실행 경계가 코드 구조로 분리된다.
- CLI는 malformed URL/dialect를 configuration/exit 2로, DBAPI package 부재, native
  import/DLL load 실패, SQLAlchemy engine availability 실패를 storage/exit 3으로
  분류한다.
- 동일 예외 타입이 initializer 밖에서 발생하면 storage로 잘못 분류하지 않는다.
- command의 integrity/programming/session 오류는 storage로 가장하지 않고 bounded
  `operation_failed`/exit 3으로 수렴한다. 단, 어떤 DBAPI subclass든
  `connection_invalidated=True`면 availability 판정이 우선한다.
- 기존 `engine`, `SessionLocal`, `get_db` 사용법이 유지된다.
- 기존 engine/sessionmaker option, lazy connection, URL precedence가 유지된다.
- CLI-owned runtime과 partial runtime은 모든 경로에서 정확히 한 번 dispose되고,
  application-global runtime은 process lifetime 동안 유지된다.
- cleanup availability 실패는 storage, cleanup programmer 실패는 operation failure로
  분류되며 두 경우 모두 기존 command 결과보다 우선하고 engine disposal을 재시도하지 않는다.
- typed exception과 CLI output에 initializer 입력 URL/driver, initializer가 잡은 원본
  exception 또는 그 민감정보가 남지 않는다. 호출자의 별도 active exception은 Python
  chaining 규칙의 범위로 명시한다.
- 모든 오류 출력은 bounded JSON이며 민감정보가 없다.
- Task 5와 Task 4 회귀가 새 PostgreSQL에서 zero-skip GREEN이다.
- 독립 리뷰 Spec PASS / Quality PASS와 controller cleanup `0:0`을 얻는다.

# Database Storage Initialization Boundary Design

검토 버전: 1  
작성일: 2026-08-29  
상태: 사용자 방향 승인 — 문서 검토 대기

## 1. 결정 요약

Deliverable C.5 Task 5의 key-admin CLI는 설정 오류와 저장소 초기화 오류를
서로 다른 bounded 결과로 보고해야 한다. 현재 구현은
`backend.app.db.session` 모듈 import 중 설정 로딩, DBAPI import, SQLAlchemy engine
생성이 함께 실행되므로 예외 클래스만으로 오류의 출처를 정확히 구분할 수 없다.

ParaWorks는 DB 계층에 설정과 분리된 **typed storage initializer**를 추가한다.
initializer는 이미 검증된 database URL을 입력받아 SQLAlchemy engine과 session
factory를 함께 생성하고, 이 소유 범위에서 발생한 DBAPI/driver/engine 초기화 실패만
내용 없는 전용 예외로 변환한다. key-admin CLI는 이 전용 예외만
`storage_unavailable`/exit 3으로 분류한다.

기존 애플리케이션의 `engine`, `SessionLocal`, `get_db` 공개 사용법은 유지한다.
Task 5 외 기능의 lazy-session 전환이나 전역 DB lifecycle 재설계는 하지 않는다.

## 2. 검토한 접근

### A. DB-owned typed initializer — 선택

- 새 DB 모듈이 URL → engine/session factory 생성만 소유한다.
- 설정 로딩이나 CLI parsing은 모듈 밖에서 완료한다.
- 저장소 초기화 실패는 전용 typed exception으로만 전달한다.
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

공개 계약은 다음 두 개다.

- `DatabaseRuntime`
  - 생성된 SQLAlchemy `Engine`
  - 해당 engine에 bind된 `sessionmaker[Session]`
- `initialize_database_runtime(database_url: str) -> DatabaseRuntime`

전용 예외 `DatabaseInitializationError`는 고정 code만 가진다. 예외 문자열이나
attribute에 database URL, username, password, driver/module 이름, 원본 exception
문구를 보존하지 않는다.

initializer가 소유하는 작업은 다음으로 제한한다.

1. 전달받은 URL로 SQLAlchemy engine 생성
2. DB dialect/DBAPI 로딩
3. engine에 bind된 session factory 생성

`Settings`, 환경변수, CLI 인자, 애플리케이션 상태는 읽지 않는다. 이 경계 안의
SQLAlchemy 초기화 오류, DBAPI import/load 오류(`ModuleNotFoundError`, `ImportError`,
native loader `OSError`)만 `DatabaseInitializationError`로 변환한다. 전용 예외는
`raise ... from None`으로 경계를 넘으며 원본 민감 출력은 폐기한다.

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

### 3.3 key-admin CLI

`backend/app/admin/auto_review_keys.py`는 `SessionLocal`을 import해 초기화하지 않는다.
이미 생성한 `Settings`에서 resolved URL을 얻은 뒤 DB initializer를 직접 호출하고,
반환된 session factory를 admin service에 주입한다.

오류 분류는 단계별로 고정한다.

- parser, Settings, 비저장소 import/초기화, 알 수 없는 failure:
  `{"ok":false,"code":"configuration_refused"}`, exit 2
- allowlisted `AutoReviewKeyAdminError`/`AutoReviewKeyBootstrapError`:
  해당 bounded code, exit 2
- `DatabaseInitializationError` 또는 command 실행 중 `SQLAlchemyError`:
  `{"ok":false,"code":"storage_unavailable"}`, exit 3
- 성공:
  기존 command별 aggregate JSON과 readiness 기반 exit code

모든 실패 출력은 stdout 한 줄의 두 필드 JSON만 사용한다. stderr, argparse usage,
traceback, local path, URL, driver name, 사용자 입력값, secret은 출력하지 않는다.

## 4. 데이터 흐름

```text
argv
  -> bounded parser
  -> Settings validation
  -> resolve database URL
  -> initialize_database_runtime(url)
       -> DBAPI/dialect/engine/sessionmaker only
       -> DatabaseRuntime OR DatabaseInitializationError
  -> AutoReviewKeyAdminService(session_factory)
  -> status/bootstrap/rebuild/rotate
  -> allowlisted JSON + exact exit code
```

비저장소 코드가 `ModuleNotFoundError`를 발생시키더라도 initializer 밖이면 설정 오류로
분류된다. 반대로 initializer 내부의 native DBAPI `ImportError`/`OSError`는 예외 타입을
추측해서가 아니라 DB 계층이 소유한 실행 단계에서 발생했으므로 storage 오류가 된다.

## 5. 테스트 전략

TDD 순서는 다음과 같다.

1. 새 initializer 단위 RED
   - 정상 URL에서 engine/session factory 생성
   - 없는 DBAPI package
   - native DBAPI `ImportError`
   - native loader `OSError`
   - typed exception에 URL/driver/original message가 없음
2. CLI subprocess RED
   - 위 세 storage 초기화 실패가 정확히 exit 3
   - parser/Settings/비저장소 `ModuleNotFoundError`는 exit 2
   - stdout exact JSON, stderr empty, 민감 sentinel 비노출
3. import/compatibility RED
   - 기존 `engine`, `SessionLocal`, `get_db` import 계약
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
- 후속 작업자에게 필요한 경우 session handoff와 portfolio log

Task 6, Slack, connector, RAG serving, application-wide lazy DB lifecycle은 제외한다.

## 7. 수용 기준

- DB 초기화와 설정/모듈 실행 경계가 코드 구조로 분리된다.
- CLI는 DBAPI package 부재, native import/DLL load 실패, SQLAlchemy engine 초기화를
  모두 storage/exit 3으로 분류한다.
- 동일 예외 타입이 initializer 밖에서 발생하면 storage로 잘못 분류하지 않는다.
- 기존 `engine`, `SessionLocal`, `get_db` 사용법이 유지된다.
- 모든 오류 출력은 bounded JSON이며 민감정보가 없다.
- Task 5와 Task 4 회귀가 새 PostgreSQL에서 zero-skip GREEN이다.
- 독립 리뷰 Spec PASS / Quality PASS와 controller cleanup `0:0`을 얻는다.


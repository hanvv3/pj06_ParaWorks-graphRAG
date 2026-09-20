# D Core 기능 기준선과 보류한 릴리스 설계

상태: **기능 우선·정식 릴리스 분리 승인 반영**. 점검 기준 `116347d`; 제품 코드 변경 없음.
기능 기준선은 새 검증 전이며 정식 release는 부분 구현/NOT CLEAN이다.
[공통 계약](2026-09-20-remaining-deliverables-design.md)과
[구현 계획](../plans/2026-09-20-d-core-completion.md)을 함께 본다.

## 지금 완료할 결과: 기능 기준선

현재 RAG V2를 다시 작성하지 않고 E가 확장할 검색/답변 경계를 검증한다.
F-1에서 실제 graph/API/Assistant와 fake provider를 연결한 회귀를, F-2에서 실제 disposable
PostgreSQL+pgvector의 권한·현재 근거·비용·transaction 경계를 확인한다. 실패한 부분만 TDD로 보완한다.

- 정상 답변·근거 없음·권한 거부·잘못된 citation·전송 전/생성 후 근거 철회가 기대한 결과로 닫힌다.
- 선택 citation뿐 아니라 전체 model influence를 재검증하고 새 AgentRun/audit를 정확히 기록한다.
- 비용 claim·실제 usage/unknown reserve·중복 dispatch 거부·provider safety/readiness를 유지한다.
- `/ask`, `/search`, Assistant 및 disabled 경로가 보존되고 SQLite smoke와 실제 PG 증거가 구별된다.
- 관련 UI build/테스트와 로컬 fake end-to-end 결과를 남긴다. 외부 서비스·실제 키·유료 호출은 쓰지 않는다.

두 작업을 통과하면 **D 기능 기준선 통과 / E 착수 가능**으로 표시한다.
이는 실제 모델 답변 품질·상용 운영·D release green을 증명하지 않는다.
기존 runtime 조립 `agent_runtime/rag_v2_composition.py`는 release reviewer/runner의 승인을
조회하는 경로가 아니다. 릴리스 gate 보류가 runtime 안전장치 비활성화를 허용하지 않는다.

## 재사용할 코드와 남은 연결

| 책임 | 현재 코드 | 남은 일 |
|---|---|---|
| 검색/답변 | `rag/retrieval.py`, `agent_runtime/rag_graph.py` | F-1: graph/API/Assistant 실제 조립 회귀 |
| runtime/노출 | `agent_runtime/rag_v2_composition.py`, `rag_finalization.py`, `rag_cost_ledger.py` | F-2: 실제 PG 권한·freshness·비용·경쟁 검증 |
| 승인·source | `rag/release_review.py`, `admin/rag_live_gate.py` | 실제 snapshot/oracle/roster reader 연결 |
| durable 상태/비용 | `rag/release_schema.py`, `release_ledger.py`, `release_authority.py` | 기존 소유권/claim/terminal 전이로 실행 제어 |
| 품질 | `rag/release_quality.py` | 승인된 결과·review 서명 결합 및 단일 publication |
| 실행 진입 | `rag/live_gate.py` | memory authority 한계 정리, bounded runner 구현 |
| 검증 | 기존 focused tests, `scripts/backend_release_matrix.py` | 지금은 F-1/F-2, formal profile 확장은 후속 D-5 |

위 상대 코드 경로의 기준은 `backend/app/`다. 정확한 타입·literal은 현재 코드/fixture를 사용한다.
기존 Task25 provider-incident 선행 수정은 별도 CLEAN 기록이 있지만 전체 release CLEAN은 아니다.

## 보류한 정식 release 계약

아래는 후속 출시 준비에서 재개할 기존 D-0~D-5의 계약이다. 지금의 E/D.1 선행 조건이 아니다.
기존 memory capability P1, preview reader/runner/reviewer/publication 연결은 미해결·보류로 남긴다.
미완성 formal CLI의 refusal/비활성 상태와 이력·테스트를 보존한다. 우회 실행이나 성공 표시를 하지 않는다.
아래 숫자는 **정식 30-case gate** 범위이며 모든 프로토타입 요청에 새 예산으로 적용하는 값이 아니다.

### 실행·품질 권한

R1 확정 시 기존 ledger의 authorization/owner/fence를 단일 실행 기준으로 사용한다.
메모리 handle 자체의 비밀성으로 승인 여부를 판단하지 않는다. 외부 DTO/HMAC 주장만으로
approval을 만들지 않고 승인된 fixture/source/corpus/provider/ledger snapshot을 조회·비교한다.

- 첫 claim·component claim은 동일 validation DB의 runtime/release/provider 상태와 함께 검증한다.
- provider 호출 전 claim을 commit한다. 네트워크 호출 동안 DB row lock을 잡아두지 않는다.
- crash 후 호출 여부가 불명확하면 reserve/unknown을 보존하고 자동 resume/재호출하지 않는다.
- 품질 평가는 허가된 30 terminal 결과와 reviewer 서명을 입력으로 받는다. 계산만으로 release가 되지 않는다.
- 보고서 insert와 authorization terminal 전이를 같은 transaction에서 완료한다.
  중복·다중 worker publication은 승인별 unique identity/CAS로 하나만 성공한다.
- 평가 도중 source/provider/corpus가 바뀌면 결과 발행 직전에 검증하여 abort한다.
  긴 사람 검토 중 snapshot/publication transaction과 row lock을 유지하지 않는다.
  다만 기존 승인 계약의 **실행자 전용 session/owner lock**은 adjudication·final까지 유지한다.
  짧은 상태 변경 transaction과 이 실행 소유권을 구별하며, crash 시 기존 attestation/fence를 따른다.

### 유지할 제한

- frozen fixture: `backend/tests/fixtures/rag_v2_live_gate_30.json`.
- 30 cases, generation 최대 30, embedding 최대 10, 전체 dispatch 최대 40, retry 0.
- case reserve USD `0.012000`, 전체 reserve USD `0.360000`.
  provider overrun actual을 clamp하지 않고 기록·중단한다. 실패 경로는 더 적은 호출로 닫힐 수 있다.
- 모델·token reserve·정확한 cost/identity는 기존 fixture/router의 승인 값을 재사용한다.
- 현재 세 reviewer 역할·서명·fresh OAuth 증명 정책을 유지한다. 모델 judge로 대체하지 않는다.
- 품질 기준: hard-negative/positive coverage 100%, required-slot 충족, faithfulness ≥95%,
  retrieval precision/recall이 고정 legacy보다 낮지 않음, 누출/무효 citation 0.

### 저장·복구

현재 release table 여섯 개는 application Base.metadata/Alembic 밖의 release 전용 metadata다.
runtime/provider와 **같은 validation PostgreSQL DB·Connection·transaction**을 쓴다.
독립 metadata를 별도 DB라는 뜻으로 해석하지 않는다. 새 테이블이 필요하면 schema version,
승인 identity, init/read/downgrade 영향을 먼저 명시한다. application Alembic에 즉시 추가하지 않는다.

external marker와 DB는 단일 원자 commit이 아니다. 현재 marker-first 후 DB 실패는
marker가 앞선 fail-stop 상태이며 자동으로 marker를 되감지 않는다. 기존 reviewed recovery를 유지한다.
v1 rebind의 유실된 이전 정책 재료를 추정하거나 새 HMAC으로 재승인하지 않는다.

### 정식 완료 증거와 공통 rollback

fake happy/failure/crash/경쟁 경로 → 실제 disposable PostgreSQL 동작 → API/Assistant 회귀 →
zero-call preview → 별도 live 승인 → 평가 결과 순으로 증거를 만든다.
PG DSN/서비스가 없으면 미실행 gate로 남기고 SQLite 통과로 대체하지 않는다.
대상 코드·fixture가 바뀌지 않은 docs-only commit에는 전체 회귀 재실행 대신 identity를 다시 확인한다.

권한/citation 문제면 해당 surface를 fail-closed/disabled로 전환한다.
AgentRun/audit/서명된 Assistant row를 지우거나 legacy 형태로 바꾸지 않는다.
기능 기준선 통과 뒤 E/D.1/Slack 개발은 허용된 순서로 진행하되 이를 정식 release 완료라 하지 않는다.
미해결 runtime provider rebind/readiness를 우회하여 유료 실험을 열지 않는다.

## 기존 계획과의 대응

F-1/F-2는 기존 Tasks1~22의 기능·DB 증거를 확인하고 필요한 실제 결함만 수정한다.
보류된 D-1~D-4는 Tasks23/24의 open gate와 Task25 잔여 연결을, D-5는 Tasks26/27을 담당한다.
기존 Tasks1~22는 필요 regression의
대상이며 체크박스가 미체크라는 이유로 재구현하지 않는다.
R1은 현재 capability 계약 변경안이므로 D-0 결정 전 관련 P1을 닫지 않는다.

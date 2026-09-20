# D Core 기능 기준선 구현·검증 계획

**Goal:** 기존 D 기능의 검증 기준선을 확보해 E GraphRAG로 진행한다. 정식 release 완성은 후속이다.
**Spec:** [D 마무리](../specs/2026-09-20-d-core-completion-design.md), [공통 계약](../specs/2026-09-20-remaining-deliverables-design.md).
**Architecture:** 현재 runtime/graph/API/Assistant와 비용·근거 경계를 검증하고 필요한 부분만 보완한다.
**Tech:** 저장소 lock의 LangChain/LangGraph, SQLAlchemy, PostgreSQL/pgvector, pytest, 기존 CLI.
**진행 방식:** 기존 subagent-driven 선택 유지. 작업별 검증·필요한 TDD 보완·영향 리뷰·커밋.
**현재:** 순서/완료 기준 분리는 승인됨. F-1/F-2는 미실행; 정식 D-0~D-5는 보류/미완료.
제품 구현은 이 문서 수정에서 시작하지 않았다. Tasks1~22 전체를 다시 시작하지 않는다.

## F-1 — 기존 기능의 provider-free 기준선 (다음 개발·검증 작업)

**파일 책임:** 기존 `backend/app/agent_runtime/rag_graph.py`, `rag_application.py`,
`rag_v2_composition.py`, `rag_finalization.py`, 관련 RAG/API/Assistant 테스트.
**입출력:** 고정 fixture + fake provider + 실제 graph/facade → 응답·citation·AgentRun/audit·비용 증거.
새 public 계약이나 별도 데모 실행 엔진을 만들지 않는다.

- [ ] HEAD/dirty 상태·테스트별 외부 호출 차단을 확인하고 기존 graph/default-runtime/API/golden 회귀를 실행한다.
  우선 명령: `uv run --no-cache --locked pytest backend/tests/test_rag_v2_graph.py backend/tests/test_rag_default_runtime.py backend/tests/test_rag_api_delivery.py backend/tests/test_rag_v2_provider_free_golden.py backend/tests/test_rag_v2_sqlite_smoke.py -q`.
- [ ] 정상·no-match·권한 거부·citation 변조·전송 전/생성 후 revoke·전체 influence·비용 보존·중복 호출의
  실제 selector coverage를 확인한다. 누락은 해당 기존 테스트에 RED를 추가하고 제품 코드의 최소 변경으로 GREEN을 만든다.
- [ ] Assistant 경계는 `test_assistant_task18.py`, 비용/최종 노출은 `test_rag_v2_finalization.py`,
  `test_rag_v2_costs.py`, `test_rag_v2_provider_safety.py`의 직접 영향 사례로 보완한다.
  실제 PG를 요구하는 사례는 F-2에 명시하고 fake 성공으로 대신하지 않는다.
- [ ] API/router가 formal `live_gate`를 우회하도록 바꾸지 않는다. 검증 revision·명령·실패/skip·수정 사항을
  portfolio 한 항목에 남기고 직접 영향 리뷰를 받는다. 제품 변경이 없으면 검증 기록만 커밋해도 된다.

**Acceptance:** 테스트한 기능 경로와 미검증 경로가 구별되고 필수 provider-free 회귀가 통과함.
과거 숫자를 재사용하지 않는다. 이 단계만으로 D 기능 기준선 전체/실제 모델 품질을 통과라 하지 않는다.

## F-2 — 실제 PG/pgvector와 로컬 통합 기준선

**파일 책임:** runtime DB binding/cost/finalization/index readiness 및 기존 integration tests.
**입출력:** 소유권을 확인한 disposable test DB + fake provider → 실제 DB·API·Assistant 통합 증거.

- [ ] 기존 테스트의 isolation/DSN 계약으로 PG+pgvector와 migration/runtime role을 준비한다.
  사용자/운영 DB를 reset하거나 schema 권한을 넓히지 않는다. 환경이 없으면 F-2 미실행으로 남긴다.
- [ ] `test_rag_v2_costs_postgres.py`, `test_rag_v2_serving_locks.py`, `test_rag_postgres_binding.py`,
  `test_rag_v2_index_readiness.py`, `test_rag_v2_pgvector_retriever.py`, `test_pgvector_integration.py`에서
  실제 PG 동작 사례를 실행하고 fake/SQLite 사례 및 skip을 구별한다.
- [ ] migration/비특권 role·동일 DB transaction·권한 필터·근거 철회·비용 claim/동시 호출을 확인한다.
  실패 원인을 재현한 뒤 관련 runtime 코드만 보완한다. provider safety 관련 변경이면 해당 실제 PG 테스트도 포함한다.
- [ ] 기존 frontend type/lint/build 및 관련 RAG/Assistant UI 테스트와 로컬 fake end-to-end를 검증한다.
  source/권한/근거/비용을 유지한 정상·실패·disabled 경로를 기록하고 remote/live rollout은 하지 않는다.
- [ ] 정확한 code revision, 환경·명령·결과·미실행 gate를 한 기록에 모아
  **D 기능 기준선 통과 / E-1 착수 가능 / 정식 release 보류·NOT CLEAN**으로 표시한다.

**Acceptance:** F-1과 실제 PG·관련 UI 통합 증거가 모두 있고 기본 안전장치를 약화한 변경이 없음.
정식 reviewer OAuth·release report publication·30-case 유료 gate 통과는 F-2의 조건이 아니다.

## 보류한 정식 release 작업

아래 D-0~D-5는 식별자와 계약을 보존한 후속 출시 준비 backlog다. F-1/F-2→E/D.1/Slack을 막지 않는다.
정식 release를 재개할 때 당시 코드를 대조하고 D-0에서 미해결 신뢰 모델을 결정한다.
F-1/F-2 성공을 capability P1 해결이나 기존 live authorization 발급으로 해석하지 않는다.

### D-0 — 계약 변경 확정 (보류, planning)

- [ ] 공통 spec R1의 trusted-process 모델과 DB 기반 durable 실행/publication을 확정한다.
- [ ] 단일성은 scoring 함수 호출이 아니라 provider dispatch와 report publication에 적용함을 확정한다.
- [ ] 기존 in-process 임의 코드 probe를 보안 경계 밖으로 분류하는 이유를 기록한다.
  해당 테스트를 단순 삭제해 통과로 만들지 않는다. 외부 DTO·source·권한 변조 거부는 유지한다.
- [ ] 기존 여섯 release table로 표현되지 않는 실제 gap이 있으면 그 한 건의 schema delta만 제안한다.

**산출물:** 이 절의 짧은 결정 기록과 영향받는 계약/테스트 목록. 정식 release 계약 변경 전 완료한다.
고립된 비신뢰 실행을 요구한다면 별도 authority 서비스 설계로 범위를 다시 산정한다.

### D-1 — durable ownership과 품질 발행 경계 (보류)

**파일 책임:** `backend/app/rag/live_gate.py`, `release_ledger.py`, `release_authority.py`,
`release_quality.py`; 필요할 때만 `release_schema.py`.
**입출력:** 기존 `AuthorizedRagLiveGate`·approved source·현재 DB snapshot → 검증된 실행/발행 문맥.
정확한 signature는 기존 코드에서 가져오며 public API는 확장하지 않는다.

- [ ] RED: 정상 1회, 두 worker 경쟁, 재시작·복제된 요청의 중복 dispatch/publication,
  cross-ledger·co-mutated DTO·stale source/provider/corpus 거부를 저장소 수준으로 재현한다.
- [ ] 기존 authorization owner/fence/state·dispatch claim·unique report를 연결한다.
  메모리 registry의 `used`를 durable 권한으로 쓰지 않는다.
- [ ] 정상/unknown/crash 비용 보존과 marker-first 실패의 fail-stop 동작을 검증한다.
  schema가 바뀌면 기존 init/read/upgrade/downgrade·승인 identity와 PG 권한 영향을 함께 검증한다.
- [ ] GREEN과 독립 리뷰 뒤 커밋. 기존 P1의 해결 방식(계약 수정/코드 수정)을 각각 기록한다.

**Acceptance:** 둘이 동시에 시도해도 승인된 효과는 하나; 재시작 후 자동 유료 재실행 없음.
**선택 테스트:** `test_rag_live_gate.py`, `test_rag_release_quality.py`,
`test_rag_release_authority_postgres.py`; schema 변경 시 `test_rag_live_gate_schema.py` 추가.

### D-2 — 실제 zero-call preview reader (보류)

**파일 책임:** `backend/app/rag/release_review.py`, `backend/app/admin/rag_live_gate.py`.
크기가 커지면 reader를 작은 별도 모듈로 추출하되 테스트된 canonicalizer는 재사용한다.
**입출력:** validation DB/source/fixture/코드·reviewer 설정 → frozen preview.

- [ ] RED: 실제 local DB에서 일관된 snapshot을 만들고, 누락/변조/dirty source/identity drift는 거부한다.
- [ ] corpus/provider/release snapshot, hard-negative oracle, reviewer roster를 실제 읽기 경로로 연결한다.
  preview는 OAuth/provider 호출과 authorization 발급을 수행하지 않는다.
- [ ] 정상 preview가 fixture·commit·DB·plan reference와 30/10/40·USD `0.360000`을 결합하는지 검증한다.
  기존 Git 환경/index/CRLF 방어의 회귀를 유지하고 커밋한다.

**Acceptance:** 정상 fixture에서 성공 preview; 실제 데이터 준비가 없으면 구체적 bounded refusal.
**선택 테스트:** `test_rag_live_gate_preview.py`, `_preview_r1.py`, `_preview_r2.py`, `_cli.py`.

### D-3 — bounded 30-case runner (보류)

**파일 책임:** `backend/app/rag/live_gate.py`, 기존 runtime admission/finalization;
`backend/tests/test_rag_live_gate.py`, 새 `backend/tests/test_rag_live_gate_postgres.py`.
**입출력:** 검증된 single-use authorization → 기존 graph 실행·sanitized terminal results.

- [ ] RED: fake happy path 30 cases/최대40 dispatch; 31번째 case, 중복 runner,
  embedding11/generation31, 자동 retry는 호출 전에 거부한다.
- [ ] case/component claim을 runtime 비용과 같은 physical DB connection/transaction에 연결한다.
  commit 이후 한 번만 dispatch하고 provider 중 row/snapshot lock을 유지하지 않는다.
  실행자 전용 session/owner lock은 기존 계약대로 최종 adjudication까지 보존해 second runner를 막는다.
- [ ] 실패 경계를 phase별로 검증한다: claim 전, claim 뒤/send 전, send 뒤/결과 전,
  finalization 전, 마지막 case 뒤. unknown 비용 보존, corpus/provider abort 우선순위가 유지돼야 한다.
- [ ] fake transport로 결과를 끝까지 만들고 PG 경쟁/연결 identity 테스트 후 커밋한다.

**Acceptance:** happy는 정확한 승인 실행, failure는 더 적은 호출로 명시적 terminal/abort;
동일 승인으로 missing downstream call을 보충하거나 resume하지 않는다.

### D-4 — 인증된 review와 quality publication (보류)

**파일 책임:** `release_quality.py`, `release_review.py`, `release_ledger.py`, `live_gate.py`,
`backend/app/admin/rag_live_gate.py`의 orchestration. CLI에 business logic을 복사하지 않는다.
**입출력:** terminal cases·일시적 review blocks·인증된 reviewer labels → 단일 report + terminal authorization.

- [ ] RED: A→B→불일치 때 C, 다른 subject/role 바꾸기·누락 서명 거부, metric red,
  중복 발행·발행 중 실패·검토 중 drift를 재현한다.
- [ ] 기존 fresh Google OAuth/PKCE proof를 재사용한다. 자동 테스트는 fake client만 사용한다.
  원문은 인증된 검토 세션에서만 보여주고 DB/report/log에는 HMAC·label만 둔다.
- [ ] `authorization-bootstrap` CLI를 실제 approved envelope·fresh reviewer proof·Task24 authorizer와
  durable bootstrap에 연결한다. 아직 preview refusal로 연결된 진입점을 교체하고,
  정상 zero-dispatch 발급과 nonce 재사용·preview drift 거부를 검증한다.
- [ ] report insert/authorization terminal 전이를 동일 transaction으로 마감한다.
  persistence 실패 후 성공으로 보고하거나 반쪽 report를 공개하지 않는다.
- [ ] `run`과 명시적 crash-abort CLI를 연결하고 기본 자동 profile에서 paid 경로가 실행되지 않음을 검증한다.

**Acceptance:** 지표 green만 quality green; ordinary failure/quality failure/provider abort가 구분됨.
**선택 테스트:** quality/live_gate/authorization/CLI 및 실제 PostgreSQL publication 경쟁.

### D-5 — 통합 검증과 실행 handoff (보류, 릴리스 준비)

**파일 책임:** `scripts/backend_release_matrix.py`, `backend/tests/release_contracts.py`,
관련 manifest 테스트와 `plan.md`/handoff/portfolio의 현재 결과.

- [ ] `rag-v2-provider-free` profile에 D 모듈과 golden ≥60의 실제 selector coverage를 추가한다.
  기존 postgres/non-slack/full profile을 유지하고 Slack baseline 변경은 S-1/S-2 수정 증거와 대조한다.
- [ ] provider-free, disposable PG+pgvector, non-Slack 통합을 검증한다.
  Task18의 실제 PG 역할/trigger/concurrency, Tasks23/24 authority, v1 rebind recovery도 추적한다.
- [ ] frontend type/lint/build와 기존 desktop/mobile RAG·Assistant·smoke E2E, secret/lock 검사를 수행한다.
- [ ] 검증 revision·명령·실패/skip·잔여 gate를 한 evidence 항목으로 기록한다.
  docs-only commit은 제품 diff가 없음을 확인하고 최종 identity/preview만 새로 검증한다.
- [ ] 정확한 clean commit의 zero-call preview를 제시한다. 별도 실행 확인 전 live를 시작하지 않는다.
  실행이 승인되어 기존 품질 gate를 통과한 뒤 D Core release green으로 표시한다.
  이는 이미 분리된 E/D.1 기능 개발의 착수 조건이 아니다.

### 정식 release 실행 명령과 검토 기준

focused 예: `uv run --no-cache --locked pytest backend/tests/test_rag_live_gate.py backend/tests/test_rag_release_quality.py -q`.
통합 예: `uv run --no-cache --locked python scripts/backend_release_matrix.py --profile rag-v2-provider-free`
(D-5가 추가한 뒤 사용). PG/non-slack은 같은 runner의 해당 profile을 사용한다.
Ruff/format은 변경 파일을, 전체 정적 검사와 frontend는 D-5에서 확인한다.

review는 external-input authority, snapshot drift, durable duplicate/crash, 비용·private data,
실제 DB 동작을 우선한다. source/fixture/의존성 변경 시 관련 증거를 다시 만든다.
유료 호출 실패를 승인 없이 실제 API로 재현하지 않는다.

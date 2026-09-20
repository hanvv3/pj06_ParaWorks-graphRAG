# D Core 마무리 설계

상태: **남은 작업 재계획안**. 코드 `3f6cb0f`는 부분 구현/NOT CLEAN.
[공통 계약](2026-09-20-remaining-deliverables-design.md)과
[구현 계획](../plans/2026-09-20-d-core-completion.md)을 함께 본다.

## 끝냈다고 말할 수 있는 결과

현재 RAG V2를 승인된 실행으로 평가하고, 근거·권한·실제 호출/비용·품질 결과를 일관되게 남긴다.
provider-free 통합 증거와 실제 PostgreSQL 증거를 확보한 뒤 zero-call preview를 제공한다.
별도 승인된 live gate까지 통과해야 D Core release green이다. rollout은 그 다음 운영 결정이다.

## 재사용할 코드와 남은 연결

| 책임 | 현재 코드 | 남은 일 |
|---|---|---|
| 검색/답변 | `rag/retrieval.py`, `agent_runtime/rag_graph.py` | 다시 작성하지 않고 runner와 연결 |
| 승인·source | `rag/release_review.py`, `admin/rag_live_gate.py` | 실제 snapshot/oracle/roster reader 연결 |
| durable 상태/비용 | `rag/release_schema.py`, `release_ledger.py`, `release_authority.py` | 기존 소유권/claim/terminal 전이로 실행 제어 |
| 품질 | `rag/release_quality.py` | 승인된 결과·review 서명 결합 및 단일 publication |
| 실행 진입 | `rag/live_gate.py` | memory authority 한계 정리, bounded runner 구현 |
| 검증 | `scripts/backend_release_matrix.py` | D 모듈 coverage/PG/프런트엔드 완료 증거 |

위 상대 코드 경로의 기준은 `backend/app/`다. 정확한 타입·literal은 현재 코드/fixture를 사용한다.
기존 Task25 provider-incident 선행 수정은 별도 CLEAN 기록이 있지만 전체 release CLEAN은 아니다.

## 실행·품질 권한

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

## 유지할 제한

- frozen fixture: `backend/tests/fixtures/rag_v2_live_gate_30.json`.
- 30 cases, generation 최대 30, embedding 최대 10, 전체 dispatch 최대 40, retry 0.
- case reserve USD `0.012000`, 전체 reserve USD `0.360000`.
  provider overrun actual을 clamp하지 않고 기록·중단한다. 실패 경로는 더 적은 호출로 닫힐 수 있다.
- 모델·token reserve·정확한 cost/identity는 기존 fixture/router의 승인 값을 재사용한다.
- 현재 세 reviewer 역할·서명·fresh OAuth 증명 정책을 유지한다. 모델 judge로 대체하지 않는다.
- 품질 기준: hard-negative/positive coverage 100%, required-slot 충족, faithfulness ≥95%,
  retrieval precision/recall이 고정 legacy보다 낮지 않음, 누출/무효 citation 0.

## 저장·복구

현재 release table 여섯 개는 application Base.metadata/Alembic 밖의 release 전용 metadata다.
runtime/provider와 **같은 validation PostgreSQL DB·Connection·transaction**을 쓴다.
독립 metadata를 별도 DB라는 뜻으로 해석하지 않는다. 새 테이블이 필요하면 schema version,
승인 identity, init/read/downgrade 영향을 먼저 명시한다. application Alembic에 즉시 추가하지 않는다.

external marker와 DB는 단일 원자 commit이 아니다. 현재 marker-first 후 DB 실패는
marker가 앞선 fail-stop 상태이며 자동으로 marker를 되감지 않는다. 기존 reviewed recovery를 유지한다.
v1 rebind의 유실된 이전 정책 재료를 추정하거나 새 HMAC으로 재승인하지 않는다.

## 완료 증거와 rollback

fake happy/failure/crash/경쟁 경로 → 실제 disposable PostgreSQL 동작 → API/Assistant 회귀 →
zero-call preview → 별도 live 승인 → 평가 결과 순으로 증거를 만든다.
PG DSN/서비스가 없으면 미실행 gate로 남기고 SQLite 통과로 대체하지 않는다.
대상 코드·fixture가 바뀌지 않은 docs-only commit에는 전체 회귀 재실행 대신 identity를 다시 확인한다.

권한/citation 문제면 해당 surface를 fail-closed/disabled로 전환한다.
AgentRun/audit/서명된 Assistant row를 지우거나 legacy 형태로 바꾸지 않는다.
D Core 완료 전 D.1/E 실행·Slack 복구·새 유료 gate를 끼워 넣지 않는다.

## 기존 계획과의 대응

새 D-1~D-4는 기존 Tasks23/24의 open gate와 Task25 잔여 연결을 마무리한다.
D-5는 Tasks26/27을 한 통합 체크포인트로 정리한다. 기존 Tasks1~22는 필요 regression의
대상이며 체크박스가 미체크라는 이유로 재구현하지 않는다.
R1은 현재 capability 계약 변경안이므로 D-0 결정 전 관련 P1을 닫지 않는다.

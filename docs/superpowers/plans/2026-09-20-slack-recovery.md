# Slack 복구 구현 계획

> **실행 방식:** `superpowers:subagent-driven-development`로 slice별 구현·리뷰한다.
> 각 slice는 RED 확인 → 최소 구현 → GREEN/영향 회귀 → 리뷰·커밋 순서로 진행한다.

**Goal:** 합성 Slack 대화가 기존 수집·검토·승인·검색 경계를 통과하는 로컬 데모를 완성한다.
**Spec:** [Slack 설계](../specs/2026-09-20-slack-recovery-design.md).
**Architecture:** fake Slack → SourceEvent/registry/shared sync → pending Review → 현재 승인 → indexing/search.
**Tech:** 기존 Python/pytest·LangChain/LangGraph·SQLite smoke·PostgreSQL/pgvector·E/D.1 경계.
**상태:** S-1/S-2 CLEAN; S-3 합성 통합 구현·독립 리뷰 대기. 다음은 정식 출시 준비 계획 결정이다.

## 공통 제약과 리뷰 초점

- [공통 계약](../specs/2026-09-20-remaining-deliverables-design.md)을 유지한다. 현재 코드가 내부 API 기준이다.
- 합성 데이터는 실제 이력으로 표시하지 않는다. 테스트의 Slack/LLM/embedding 외부 호출은 0이다.
- 권한·Review trust·source authority 변경은 계획 작성만으로 승인되지 않는다. unsigned legacy 행은 fail-closed다.
- 첫 slice 착수 시 현재 HEAD·실패 node id·fixture·관련 테스트 명령을 고정한다. SDK·새 schema는 미리 만들지 않는다.
- 리뷰 초점: 늦은 reply/cursor(S-1), replay count(S-1), 수정·삭제와 근거 수명(S-2),
  제한 권한·승인 우회(S-2), cache/graph 잔존 근거와 offline/live 구분(S-3).

## S-1 — 합성 수집과 기존 실패 재분류

**파일 책임:** `backend/app/connectors/slack.py`, `registry.py`, `ingestion/sync.py`;
`backend/tests/test_slack_connector.py`, `test_connector_ingestion_contract.py`, 관련 합성 fixture.

- [x] 현재 `backend/tests/release_contracts.py`의 보류 Slack test id를 실행해 과거 10개 실패와 비교한다.
  실제 결과를 계약 결함/fixture 부적합/환경 부족으로 분류하고 unresolved 항목을 보존한다.
- [x] RED: channel/thread/participants/source 근거, bounded reply window, 오래된 parent의 새 reply,
  cursor 경계, 동일 이벤트 replay와 fetched/created/skipped count를 fake API로 재현한다.
- [x] 공통 connector/sync 경계 안에서 최소 수정하고 GREEN을 확인한다. pending 생성을 아직 못 하는
  legacy 경로는 S-2 의존성으로 기록하며 수집 성공만으로 전체 복구를 선언하지 않는다.
- [x] 외부 호출 0과 기존 connector 영향 회귀를 확인하고 fixture·실패 분류·코드를 리뷰·커밋한다.

**완료:** 합성 source가 식별자·문맥·근거를 보존해 재수집되고 중복 count가 설명된다.

S-1 code `1ddfc21`, [검증·한계](../runbooks/s-1-slack-synthetic-ingestion.md).
외부 attempts 0, R1 영향 테스트 101 passed; 독립 리뷰 CLEAN. 당시 10개 실패는 S3에서 수정했다.

## S-2 — 현재 Review·source lifecycle과 검색 연결

**파일 책임:** `backend/app/ingestion/service.py`, `agents/slack_agent/service.py`, 관련 runtime registry,
`review/transitions.py`, `rag/indexing.py`; Slack review bridge·knowledge promotion·indexing 테스트.

- [x] missing evidence, restricted 혼합 입력, legacy unsigned/current-version 차이,
  pending의 trusted 노출 거부와 기존 승인 전이를 거친 검색을 재현한다.
- [x] 기존 source authority/Review 계약과의 gap을 최소 범위로 연결한다. fresh ingestion과 기존 행
  migration을 구분하고 정책 변경이 필요하면 범위·영향을 먼저 명시한다. 승인 데이터를 직접 seed하지 않는다.
- [x] source 수정·삭제·권한 축소, 승인 재시도, prompt version 변경을 고정하고 최소 구현한다.
  근거 version과 strictest permission, token/cost, agent cache와 incremental embedding skip을 확인한다.
- [x] fake 모델 기반 GREEN과 Review/권한/indexing 영향 회귀를 실행하고 변경 경계를 리뷰·커밋한다.

**완료:** 수집 → pending → 현재 승인 → 검색이 연결되고, source 변경 시 오래된 지식 노출이 차단된다.

구현·fresh 검증은 [S2 runbook](../runbooks/s-2-slack-synthetic-review.md).
SQLite 영향 215 passed/실제 PG·pgvector 3 passed, external 0. `d91ed9d` 독립 리뷰 CLEAN.

## S-3 — 통합 데모와 실제 연결 인수인계

**파일 책임:** S-1 fixture의 로컬 smoke 시나리오, `scripts/backend_release_matrix.py`와
`backend/tests/release_contracts.py`의 증거 기반 기대값, portfolio/handoff의 결과 기록.

- [x] S-2의 승인 시나리오에 E 관계 검색·D.1 캐시를 연결한다. 수정/철회/권한 축소 때 graph/cache가
  과거 근거를 노출하는 실패부터 고정하고 fresh GREEN과 flag off 복귀를 확인한다.
- [x] disposable DB/fake 주입으로 수집 count, pending/approved 결과, 근거 검색, replay와 saved calls를
  보여준다. SQLite smoke와 PostgreSQL/pgvector/Neo4j 증거를 구분하고 실제 외부 호출 0을 확인한다.
- [x] 해결한 Slack node id만 검증 결과와 함께 baseline에서 갱신한다. 미해결 실패를 숨기지 않고
  backend 통합·관련 frontend 검증 결과를 release readiness로 넘긴다.
- [ ] 합성 완료/남은 실패/live 준비 상태를 코드 revision과 함께 기록하고 리뷰·커밋한다.

**완료:** 합성 end-to-end를 다시 실행할 수 있고 기존 권한·승인·근거·비용 경계가 유지된다.
실제 workspace/export/대체 source와 접근 권한은 연결 단계의 사용자 결정이다. 미선택은 합성 완료를
막지 않으며 live 준비 완료로도 표시하지 않는다. 다음 단계는 전체 release readiness다.

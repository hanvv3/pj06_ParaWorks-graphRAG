# D.1 답변 캐시 구현 계획

**Goal:** 현재 근거 검색 뒤 generation 재사용으로 비용을 줄인다.
**Spec:** [D.1 설계](../specs/2026-09-20-d1-answer-cache-design.md).
**Architecture:** 작은 cache port + PostgreSQL store + SQLite Null store, 기존 evidence/finalization 재사용.
**Tech:** 기존 SQLAlchemy/Alembic/PostgreSQL/LangGraph, fake model 테스트.
**상태:** 계획안; D Core release green 이후 C-1만 상세화해 진행한다. 세부 schema/API는 아직 고정하지 않는다.

## C-1 — 격리·보존을 보장하는 저장소

**파일 책임:** 신규 `backend/app/rag/answer_cache.py`(port/key/eligibility),
`answer_cache_store.py`(PG/Null 구현), 해당 model·forward migration·테스트.
새 파일명은 착수 시 현 코드와 맞추되 책임 분리는 유지한다.

- [ ] D Core 최종 identity/finalization과 대조해 schema·TTL(최대24h)·key/dependency를 정한다.
- [ ] RED: 같은 query라도 principal/scope/input-context/version/key가 다르면 miss,
  expiry·signed value 불일치·금지 raw field 저장 거부, SQLite no-write를 재현한다.
- [ ] validated answer blocks + 전체 influence 참조만 보관하고 TTL 정리·flag disable을 구현한다.
- [ ] PG 저장/권한/expiry 회귀와 migration/rollback 영향을 확인해 커밋한다.

**Acceptance:** 독립 저장소 테스트에서 안전한 hit/miss/expiry만 제공하며 아직 제품 답변을 재사용하지 않는다.

## C-2 — graph·권한·회계 연결

**파일 책임:** `backend/app/agent_runtime/rag_graph.py`, `rag_finalization.py`,
`backend/app/rag/evidence_projection.py`, C-1 store. 새 focused cache 통합 테스트.

- [ ] RED: fresh retrieval 뒤 valid hit은 generation0, miss는 기존 generation1;
  query embedding 실제 비용, 새 AgentRun/Audit, selected/nonselected revoke를 검증한다.
- [ ] cache-hit substantive finalization을 명시적으로 정의하고 기존 paid/canned 경로와 구별한다.
  current citation 재구성·최종 영향 근거 재검증을 공통 경로로 사용한다.
- [ ] 새 관련 근거/Assistant 문맥 변화와 lookup→publication 사이 drift를 거부한다.
  store 오류는 miss, 권한 authority 오류는 fail-closed로 처리한다.
- [ ] 직접 영향 RAG/API/Assistant 회귀 뒤 독립 리뷰·커밋한다.

**Acceptance:** hit이 공개 계약/권한/비용 기록을 약화하지 않으면서 generation 호출을 실제로 절감한다.

## C-3 — 비교와 제한 활성화 준비

- [ ] 동일 fixture의 cold/warm 호출·비용·p50/p95·DB queries·hit-rate를 기록한다.
- [ ] 다른 사용자/동일 role, TTL, policy/key 변경, drift, DB 장애, flag rollback을 확인한다.
- [ ] PG·API·Assistant 통합 및 secret 검증 후 안전·비용 성과를 handoff 한 항목에 남긴다.
  운영 flag 활성화와 유료 측정은 별도 범위로 다룬다.

**완료:** 정확한 isolation/revalidation, 새 audit와 zero-generation accounting, 측정된 절감,
기존 경로 rollback이 확인됨. embedding 절감·Redis·semantic cache는 완료 요건에 추가하지 않는다.
다음 E 구현 전 이 세 slice가 green이어야 한다.

검증은 새 cache 모듈의 실패 재현 → 수정 → 영향 회귀 순서다.
현재 존재하지 않는 test 명령/SDK API를 미리 고정하지 않고 C-1에서 실제 파일명을 정한다.

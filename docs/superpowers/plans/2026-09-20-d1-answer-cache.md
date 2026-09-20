# D.1 답변 캐시 구현 계획

> **실행 방식:** 기존 `superpowers:subagent-driven-development` 선택을 유지한다. C-1/C-2 구현 후 C-2 독립 리뷰, 이어 C-3을 진행한다.

**Goal:** 현재 근거 검색 뒤 generation 재사용으로 비용을 줄인다.
**Spec:** [D.1 설계](../specs/2026-09-20-d1-answer-cache-design.md).
**Architecture:** 작은 cache port + PostgreSQL store + SQLite Null store, D evidence/E path finalization 재사용.
**Tech:** 기존 SQLAlchemy/Alembic/PostgreSQL/LangGraph, fake model 테스트.
**상태:** C-1 저장소와 C-2 graph/회계/finalization 연결 구현. C-2 독립 리뷰 대기, C-3 미착수.
세부 schema/API는 착수 시 정한다. D release green은 선행 조건이 아니며 정식 live gate는 뒤에 남는다.

## 공통 제약과 검토 초점

공통 spec과 E의 PG relation 검증 계약을 유지한다. cache는 default disabled, SQLite는 Null store,
TTL 최대 24시간이며 fresh retrieval 뒤 generation만 재사용한다. 공개 API/citation을 바꾸지 않는다.
검토 초점은 동일 role의 다른 principal, 비선택 근거/edge revoke, 새 근거/문맥, hit 후 최종 drift,
hit substantive 비용·audit다. 아래 작업에서 각각 실패 재현과 회귀를 확인한다.

## C-1 — 격리·보존을 보장하는 저장소

**파일 책임:** 신규 `backend/app/rag/answer_cache.py`(port/key/eligibility),
`answer_cache_store.py`(PG/Null 구현), 해당 model·forward migration·테스트.
새 파일명은 착수 시 현 코드와 맞추되 책임 분리는 유지한다.

- [x] D evidence와 E-2의 versioned path carrier/finalization을 대조해 schema·TTL(최대24h)·key/dependency를 정한다.
  전체 ordered influence와 relation의 canonical identity/version·scope/permission 검증 참조를 포함한다.
  Neo4j projection generation 또는 citation endpoint만으로 유효성을 판단하지 않는다.
- [x] RED: 같은 query라도 principal/scope/input-context/version/key가 다르면 miss,
  relation 변경·backend 전환도 miss, expiry·signed value 불일치·금지 raw field 저장 거부,
  SQLite no-write를 재현한다.
- [x] validated answer blocks + 전체 influence 참조만 보관하고 TTL 정리·flag disable을 구현한다.
- [x] PG 저장/권한/expiry 회귀와 migration/rollback 영향을 확인한다.

내부 schema/consumer 계약과 검증 범위는 [C-1 runbook](../runbooks/c-1-answer-cache-storage.md)을 따른다.

**Acceptance:** 독립 저장소 테스트에서 안전한 hit/miss/expiry만 제공하며 아직 제품 답변을 재사용하지 않는다.

## C-2 — graph·권한·회계 연결

**파일 책임:** `backend/app/agent_runtime/rag_graph.py`, `rag_finalization.py`,
`backend/app/rag/evidence_projection.py`, E의 path carrier/PG validator, C-1 store.
새 focused cache 통합 테스트.

- [x] RED: fresh retrieval 뒤 valid hit은 generation0, miss는 기존 generation1;
  query embedding 실제 비용, 새 AgentRun/Audit, selected/nonselected 및 중간 node/edge revoke를 검증한다.
- [x] cache-hit substantive finalization을 명시적으로 정의하고 기존 paid/canned 경로와 구별한다.
  current citation 재구성·전체 영향 근거/relation의 PG 재검증을 공통 경로로 사용한다.
- [x] 새 관련 근거/Assistant 문맥 변화와 lookup→publication 사이 drift를 거부한다.
  endpoint 문서가 그대로인 relation version/permission 변경도 포함한다.
  store 오류는 miss, 권한 authority 오류는 fail-closed로 처리한다.
- [ ] 직접 영향 RAG/API/Assistant 회귀 뒤 독립 리뷰·커밋한다.

**Acceptance:** hit이 공개 계약/권한/비용 기록을 약화하지 않으면서 generation 호출을 실제로 절감한다.

내부 `answer-cache-hit:v1` 모드, 감사/보존/운영 경계는
[C-2 runbook](../runbooks/c-2-answer-cache-runtime.md)을 따른다.
실제 PG 조립 테스트는 E의 metadata schema/scorer fixture이며 전체 migration-trigger 검증이 아니다.

## C-3 — generation 절감 비교와 rollback

- [ ] E fixture의 같은 corpus·principal·retrieval 예산·모델 설정과 동일 backend로 cold/warm
  generation 호출·비용·p50/p95·DB queries·hit-rate를 비교한다. GraphRAG 검색 개선은 E의
  cache-off pgvector/graph 결과로, cache 절감은 이 cold/warm 결과로 따로 보고한다.
  fake 추정 비용과 실제 provider 측정은 분리하며 유료 측정은 별도 승인 범위에서만 실행한다.
- [ ] 다른 사용자/동일 role, TTL, policy/key 변경, drift, DB 장애, flag rollback을 확인한다.
  cache-off가 E fresh retrieval/generation으로, graph-off가 keyword/pgvector로 복귀하는지 검증한다.
- [ ] PG·API·Assistant 통합 및 secret 검증 후 안전·비용 성과를 handoff 한 항목에 남긴다.
  운영 flag 활성화와 유료 측정은 별도 범위로 다룬다.

**완료:** 정확한 isolation/revalidation, 새 audit와 zero-generation accounting, 측정된 절감,
기존 경로 rollback이 확인됨. embedding 절감·Redis·semantic cache는 완료 요건에 추가하지 않는다.
fake로 확인한 호출 절감은 실제 모델 품질·실비 절감 증거와 구분한다. 다음은 Slack 복구이며,
그 뒤 정식 릴리스 준비에서 D의 남은 live gate와 E/D.1 release 검증을 수행한다.

검증은 새 cache 모듈의 실패 재현 → 수정 → 영향 회귀 순서다.
현재 존재하지 않는 test 명령/SDK API를 미리 고정하지 않고 C-1에서 실제 파일명을 정한다.

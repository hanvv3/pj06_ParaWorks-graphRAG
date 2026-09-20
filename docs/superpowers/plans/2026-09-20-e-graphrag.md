# E GraphRAG 구현 계획

> **실행 방식:** 기존 `superpowers:subagent-driven-development` 선택을 유지한다. 현재 요청은 문서 수정이며 구현을 시작하지 않는다.

**Goal:** 승인된 관계와 근거를 따라 검색하는 Neo4j adapter를 기존 RAG에 연결한다.
**Spec:** [E 설계](../specs/2026-09-20-e-graphrag-design.md).
**Architecture:** PostgreSQL-derived projection → bounded graph retrieval → canonical evidence 검증 → 기존 answer graph.
**Tech:** 기존 LangChain Runnable/LangGraph, PostgreSQL, 공식 Neo4j driver; 필요한 SDK만 착수 시 lock.
**상태:** E-1 projection 완료, E-2 구현/검증 후 독립 리뷰 대기, E-3 미구현. 상세 계약과 재현은
[E-1 runbook](../runbooks/e-1-graph-projection.md)을 따른다.
정식 D release green은 선행 조건이 아니다. 순서는 E → D.1 → Slack → 정식 릴리스 준비다.

## 공통 제약과 검토 초점

공통 spec의 evidence·permission·budget·공개 API 계약을 유지한다. 캐시 없이 실제 LangChain
Runnable/LangGraph 연결을 먼저 검증하며 PostgreSQL/pgvector의 authority를 이중화하지 않는다.
검토 초점은 restricted 중간 경로, endpoint가 그대로인 relation drift, 부분 sync/삭제,
graph 장애 시 중복 embedding·budget, 도움이 없는 질문의 회귀다. 아래 작업에 각 검증을 포함한다.

## E-1 — 재실행 가능한 작은 projection

**파일 책임:** 신규 `backend/app/rag/graph_projection.py`(mapping/sync),
`graph_store.py`(driver 경계), 관련 admin job·모델/테스트.
기존 `indexing.py`, `serving_generation.py`, canonical models의 lifecycle를 재사용한다.

- [x] 구현 전에 spec의 관계/단일 근거/근거 없음/restricted/revoke 질문군을 작은 synthetic fixture로
  고른다. 기대 source ID·version·승인 연결 및 principal별 허용/금지 ID를 고정한다.
  같은 corpus·principal·질문·모델·retrieval/evidence 예산과 cache-off 조건의 pgvector baseline을 기록한다.
- [x] fixture에서 필요한 최소 explicit relation allowlist와 PG provenance/version 판정법을 정한다.
  각 관계가 어떤 canonical source/승인 연결에서 재구성되는지 명시한다. PG에서 검증할 수 없는
  관계는 제외한다. keyword/pgvector seed 재사용·seed cap·hop/candidate/time 한도도 함께 정한다.
- [x] E-2가 운반할 최소 내부 path dependency 계약을 정한다: 영향을 준 ordered node/edge/중간 근거의
  canonical identity/version·scope/permission 검증 참조. 현재 RetrievalResult에 이미 있다고 가정하지 않는다.
  구체 필드와 기존 state/fingerprint 소비자의 변경 범위를 고르고 SDK는 lock·현재 공식 문서로 확인한다.
- [x] RED: 중복 sync·crash/restart·revoke/delete/supersede·scope 충돌·stale generation을 재현한다.
- [x] stable identity/version 기반 bounded reconcile와 tombstone 처리·lag/count 관측을 구현한다.
  fake driver와 disposable Community DB로 검증했다. **실제 배포 최소 권한 RBAC는 미검증**이며
  schema admin/worker 분리 지침을 runbook에 남겼다. Community 테스트 계정을 운영 권한 증거로 쓰지 않는다.
- [x] disposable Neo4j에서 재시작/중복/삭제 수렴을 검증하고 코드·증거를 커밋한다.

**Acceptance:** 동일 입력의 재실행이 중복 graph를 만들지 않고 PostgreSQL 현재 상태로 수렴한다.
fixture 기대 근거와 PG에서 확인 가능한 relation dependency 계약이 고정돼야 E-2로 진행한다.
Neo4j가 꺼져도 기존 RAG가 동작한다. outbox/CDC와 답변 캐시는 선행 조건이 아니다.

## E-2 — 권한을 보존하는 검색 adapter

**파일 책임:** 신규 `backend/app/rag/neo4j_retriever.py`, 기존 `retrieval.py`,
`backend/app/agent_runtime/rag_v2_contracts.py`, `rag_graph.py`, `rag_finalization.py`,
`backend/app/rag/evidence_projection.py`와 graph registry/routing.
backend union·내부 path carrier·state/fingerprint version 변경의 소비자를 함께 점검한다.

- [x] RED: relation 질의의 유효 provenance, restricted 중간 node/edge, scope 경계,
  stale/deleted path, graph 장애, bounded hidden count와 중복 embedding 거부를 검증한다.
- [x] parameterized bounded traversal을 LangChain adapter로 구현하고 PostgreSQL canonical 근거를
  다시 확인한다. public citation이나 trusted status를 graph 응답에서 직접 만들지 않는다.
- [x] E-1의 versioned path dependency를 RetrievalResult → graph state → 모델 전송 전 PG 검증 →
  최종 노출 검증까지 연결한다. keyword/pgvector의 빈 dependency 호환과 fallback/disabled를 보존한다.
- [x] 캐시가 없는 경로에서 generation 도중 중간 edge revoke/version/permission 변경을 RED로 재현하고,
  전체 relation dependency를 최종 PG 검증에 포함한다. endpoint 문서 검증만으로 통과시키지 않는다.
- [ ] fake graph+실제 PG/Neo4j 계약, 기존 API/Assistant 회귀를 검증·리뷰하고 커밋한다.
  graph 장애 fallback도 seed/embedding을 재사용하고 동일 budget을 유지하며 authority 장애는 거부한다.

**Acceptance:** 버전이 명시된 내부 결과가 전체 path 검증을 보존하고 공개 route/화면은 늘리지 않는다.
구현·검증 범위와 fake/real DB 경계는 [E-2 runbook](../runbooks/e-2-relationship-retrieval.md).
마지막 검증·리뷰 항목은 독립 리뷰까지 열린 상태로 둔다.
지원하지 않는 질의와 권한/authority 오류를 graph text로 임의 보충하지 않는다.
E-2의 dependency 계약을 후속 D.1에 넘기며 아직 캐시 저장소나 hit 경로를 구현하지 않는다.

## E-3 — 관계 질의 개선과 rollback 증명

- [ ] E-1의 같은 corpus·principal·모델·예산과 cache-off 조건으로 pgvector/graph를 비교한다.
  기대 source ID 기준 precision/recall·관계 coverage와 단일 근거/no-match 회귀를 기록한다.
- [ ] fake 모델의 기능 계약과 실제 모델의 관계 답변 성공·faithfulness·citation 평가를 분리한다.
  실제 모델 평가는 별도 승인된 실행 범위에서만 수행하며 미실행이면 품질 미평가로 남긴다.
  latency·비용·projection lag도 기록하되 baseline 없는 임의 SLO나 fake 결과를 live 통과라 하지 않는다.
- [ ] graph unavailable/stale/revoke, flag rollback·PG/Neo4j 재시작을 검증한다.
  prototype 기능 결과와 향후 shadow·제한 활성화 판단을 구분하며 flag를 자동 활성화하지 않는다.
- [ ] 코드 revision/fixture/검색 결과·DB 증거·실제 모델 미평가 항목을 한 evidence 항목에 남긴다.

**Prototype 기능 완료:** cache-off 관계 검색의 이점, 현재 permission/evidence/citation 보존,
실제 PG/Neo4j 동작과 장애 복귀가 확인됨. 실제 모델 품질/정식 live 평가 미완료는 별도 남긴다.
Text2Cypher·유료 classifier·Knowledge Map 확대·CDC는 별도 필요가 입증될 때 추가한다.
다음은 D.1 캐시이며 E의 path dependency를 key/hit/finalization에 재사용한다. 그 뒤 Slack,
마지막 정식 릴리스 준비에서 남은 D live 평가와 새 기능의 release 검증을 수행한다.

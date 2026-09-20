# E — 최소 Neo4j GraphRAG 설계

상태: **방향·계약 계획안, 미구현**. D Core와 D.1 완료 뒤 착수 시 API/의존성을 확인한다.
[공통 계약](2026-09-20-remaining-deliverables-design.md) / [계획](../plans/2026-09-20-e-graphrag.md).

## 목표와 첫 사용 사례

현재 권한으로 볼 수 있는 승인 지식과 근거의 연결을 따라 관계 질문을 더 잘 검색한다.
예: 한 프로젝트의 결정과 그 결정을 뒷받침하는 문서/이력 연결.
처음에는 명시된 프로젝트 연결·승인 provenance·`supported_by` 관계를 사용한다.
단순 vector 검색 대비 개선이 없다면 운영 기본 경로로 활성화하지 않는다.

## PostgreSQL 기준의 파생 graph

Neo4j는 재구성 가능한 projection이다. canonical serving identity/version·승인 근거·권한의
기준은 PostgreSQL이다. raw source는 provenance로 유지하고 AI 추론 relation은 별도 검토 정책
없이 trusted fact로 승격하지 않는다. 관계 metadata에서 신뢰도를 추정해 승인 대신 쓰지 않는다.

- 시작은 bounded batch reconciliation: 안정된 identity, content/version hash, 진행 cursor,
  재시작 시 중복 안전한 upsert, revoke/delete/supersede 처리, projection generation 기록.
- delete도 발견할 수 있는 기존 tombstone/reconciliation 경로를 사용한다. `updated_at`만
  읽어서 삭제를 영구히 놓치는 sync는 허용하지 않는다. 재구성·재실행으로 상태를 수렴시켜야 한다.
- 현재 규모에서는 CDC/outbox/broker를 필수 선행 dependency로 두지 않는다.
- graph에는 초기 검색에 필요한 identifier·관계·scope/version만 저장한다.
  원문/embedding을 중복 복제하는 별도 벡터 저장소를 기본으로 만들지 않는다.

## 검색 경계

기존 LangChain `Runnable[RetrievalRequest, RetrievalResult]` port와 registry에 adapter를 추가한다.
현재 backend 타입이 keyword/pgvector로 제한되므로 E-2에서 버전·내부 enum·fallback/caching 영향을
명시적으로 확장한다. API route/connector가 graph를 직접 조립하지 않는다.

초기 탐색은 parameterized query와 제한된 relation allowlist·hop/후보/시간 예산으로 수행한다.
E-1에서 fixture로 유용성을 증명한 가장 작은 탐색 범위를 고정한다.
시작점은 기존 keyword/pgvector 검색의 canonical identity를 우선 사용하며 seed 수를 제한한다.
이미 구한 embedding/seed를 재사용해 graph 선택과 fallback이 추가 embedding 호출을 만들지 않게 한다.
LLM이 만드는 임의 Cypher나 새 유료 classifier는 첫 버전에 필요하지 않다.

노드뿐 아니라 edge·중간 path에도 scope/permission을 적용한다. 모델 전송 전에 PostgreSQL에서
현재 canonical 근거와 관계 provenance를 확인하고 stale/revoked/불완전한 path를 제거한다.
graph path가 답변에 영향을 주면 전체 path 근거가 D.1 dependency/version identity에 들어간다.
캐시가 꺼져 있어도 최종 응답 직전에 path의 relation/version/permission을 다시 검증한다.
endpoint 문서가 같아도 생성 도중 edge가 철회될 수 있으므로 기존 문서 검증만으로 충분하다고 보지 않는다.
hidden count는 기존 bounded candidate 의미를 유지하며 edge/path 개수를 공개하지 않는다.

graph 장애/지연 때는 독립적으로 검증한 기존 검색 경로로만 제한 fallback한다.
권한 authority를 읽지 못하는 경우는 fail-closed이며 graph 결과를 그대로 사용하지 않는다.
fallback도 query embedding 중복 호출·reserve 우회를 만들지 않아야 한다.

## 검증·배포 범위

같은 frozen fixture로 pgvector와 graph를 비교한다: 관계 질문 성공률, precision/recall,
faithfulness, p50/p95, 비용, projection lag. 정량 SLO는 baseline과 함께 고정한다.
안전 기준은 기존 D 기준을 유지한다. graph가 유용한 사례와 도움이 없는 사례를 모두 기록한다.

초기 acceptance: 중간 restricted node, 교차 scope, edge revoke, stale projection, 중복 sync,
재시작, 삭제 수렴, Neo4j 장애, D.1 cache invalidation. 실제 Neo4j 검증과 fake 계약 검증을 구분한다.
shadow 뒤 제한 활성화하고 rollback은 graph 선택 flag off로 수행한다. 원본/audit는 보존한다.
사용자 화면과 citation shape는 그대로다. Knowledge Map 확대는 별도 제품 요구가 있을 때 진행한다.

## 기존 자산과 library 선택

`backend/app/api/v1/knowledge.py`의 현재 map은 PostgreSQL 조립 결과다. Neo4j 구현 증거가 아니다.
`rag/retrieval.py`, `serving_contracts.py`, `evidence_projection.py`, `indexing.py`,
`serving_generation.py`를 경계로 재사용한다. Neo4j driver는 공식 라이브러리를 사용한다.
`neo4j-graphrag` adapter는 요구에 맞는 검색 기능을 재사용할 때 도입한다. 이를 쓰기 위해
별도 graph/vector/LLM framework를 재작성하거나 PostgreSQL embedding authority를 이중화하지 않는다.
SDK 버전·호출은 E-1에서 lock과 공식 문서로 확인한다.

공식 참고: [Neo4j GraphRAG retriever 가이드](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
(2026-09-20 조회). 문서의 범용 예시가 ParaWorks 권한·보존 계약을 대체하지 않는다.

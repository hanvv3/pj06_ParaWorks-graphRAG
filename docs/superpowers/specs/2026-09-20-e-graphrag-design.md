# E — 최소 Neo4j GraphRAG 설계

상태: **승인된 프로토타입 우선순위, 미구현**. D 기능 baseline(F-1/F-2) 뒤, D.1 캐시보다 먼저 착수한다.
[공통 계약](2026-09-20-remaining-deliverables-design.md) / [계획](../plans/2026-09-20-e-graphrag.md).

순서: `D 기능 baseline → E 최소 GraphRAG → D.1 캐시 → Slack → 정식 릴리스 준비`.
D의 fake API/Assistant 회귀와 실제 disposable PostgreSQL/pgvector 기능 계약을 먼저 확인한다.
이 baseline은 D release green이 아니다. 정식 live 평가와 남은 release gate는 뒤로 옮기며 면제하지 않는다.
이 문서는 제품 코드·provider 호출·flag 활성화 또는 유료 실행을 승인하지 않는다.

## 목표와 첫 사용 사례

현재 권한으로 볼 수 있는 승인 지식과 근거의 연결을 따라 관계 질문을 더 잘 검색한다.
예: 한 프로젝트의 결정과 그 결정을 뒷받침하는 문서/이력 연결.
처음에는 PostgreSQL에서 확인 가능한 명시된 프로젝트 연결·승인 provenance·`supported_by`
관계를 사용한다. 캐시를 끈 상태에서 단순 vector 검색 대비 검색 개선을 먼저 확인한다.
개선이 없다면 범위나 관계 가정을 재검토하며 운영 기본 경로로 활성화하지 않는다.

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

기존 LangChain `Runnable[RetrievalRequest, RetrievalResult]` port와 registry에 adapter를 추가하고
기존 LangGraph answer graph에 연결한다. 현재 backend 타입은 keyword/pgvector로 제한되어 있고,
`RetrievalResult`는 전체 path provenance를 운반하지 않는다. 동일 타입이 이미 충분하다고 가정하지 않는다.
API route/connector가 graph를 직접 조립하지 않는다.

**최소 내부 계약:** E-1에서 허용 relation을 어떤 PostgreSQL canonical source/승인 연결에서
재구성하는지와 stable identity/version 판정법을 확정한다. E-2는 검색 결과와 graph state에
버전 있는 내부 dependency carrier를 추가해 답변에 영향을 준 ordered path의 node·edge·중간 근거,
그 canonical identity/version 및 scope/permission 검증 참조를 finalization까지 운반한다.
Neo4j generation만으로 관계의 현재성을 증명하지 않는다. PG에서 관계를 재구성·검증할 수 없으면
첫 relation allowlist에서 제외한다. 기존 keyword/pgvector는 빈 path dependency로 호환한다.
backend enum·state/fingerprint 버전과 소비자 테스트를 함께 갱신하되 공개 response/citation shape는 유지한다.
구체적인 필드·저장 delta는 이 책임을 충족하는 최소 변경으로 정하며 새 authority 저장소를 만들지 않는다.

초기 탐색은 parameterized query와 제한된 relation allowlist·hop/후보/시간 예산으로 수행한다.
E-1에서 fixture로 유용성을 증명한 가장 작은 탐색 범위를 고정한다.
시작점은 기존 keyword/pgvector 검색의 canonical identity를 우선 사용하며 seed 수를 제한한다.
이미 구한 embedding/seed를 재사용해 graph 선택과 fallback이 추가 embedding 호출을 만들지 않게 한다.
LLM이 만드는 임의 Cypher나 새 유료 classifier는 첫 버전에 필요하지 않다.

노드뿐 아니라 edge·중간 path에도 scope/permission을 적용한다. 모델 전송 전에 PostgreSQL에서
현재 canonical 근거와 관계 provenance를 확인하고 stale/revoked/불완전한 path를 제거한다.
답변에 영향을 준 전체 path dependency를 모델 전송 시점과 최종 응답 직전에 PostgreSQL에서
재검증한다. 이 계약은 캐시 없이 E에서 먼저 완성한다. 후속 D.1은 동일 dependency/version
identity를 key·hit 검증에 포함한다.
endpoint 문서가 같아도 생성 도중 edge가 철회될 수 있으므로 기존 문서 검증만으로 충분하다고 보지 않는다.
hidden count는 기존 bounded candidate 의미를 유지하며 edge/path 개수를 공개하지 않는다.

graph 장애/지연 때는 독립적으로 검증한 기존 검색 경로로만 제한 fallback한다.
권한 authority를 읽지 못하는 경우는 fail-closed이며 graph 결과를 그대로 사용하지 않는다.
fallback도 query embedding 중복 호출·reserve 우회를 만들지 않아야 한다.

## 구현 전에 고를 작은 비교 fixture

E-1의 첫 산출물은 아래 질문군을 포함한 작은 synthetic corpus와 기대 근거표다.
실제 fixture의 source ID·version·승인 연결·principal별 허용/금지 ID를 고정하고 baseline을 얻은 뒤
projection 구현을 시작한다. 건수를 임의 목표로 삼지 않고 각 실패/성공 조건을 구분할 만큼만 고른다.

| 질문군 | 비교할 의미와 기대 근거 |
|---|---|
| 관계 질문 | 프로젝트의 결정 이유와 후속 문서를 연결한다. 결정·연결을 지지하는 원본·후속 문서의 기대 source ID를 명시한다. |
| 단일 근거 질문 | 한 승인 문서로 답할 수 있다. 그 source ID를 명시하고 graph가 무관한 근거를 더하지 않는지 본다. |
| 근거 없음 | corpus 밖 질문으로 기대 source ID는 빈 집합이며 근거를 만들어내지 않아야 한다. |
| restricted 경로 | 관계 질문의 중간 근거/edge 권한을 바꾼다. 같은 principal에서 제외되어야 할 source ID와 남는 허용 ID를 명시한다. |
| revoke/version 변경 | 동일 endpoint를 둔 채 관계를 철회·변경한다. 검색 후 또는 생성 중 변경도 관계 답변의 최종 노출을 막아야 한다. |

pgvector와 graph는 동일 corpus snapshot·principal·질문·모델 설정·retrieval/evidence 예산으로
비교하고 cache는 off로 둔다. graph의 별도 hop/시간 한도도 기록하며 embedding·후보 예산을
늘린 효과를 관계 검색 개선으로 세지 않는다. 기존 pgvector baseline을 같은 조건으로 보존한다.
deterministic/test vector로 비교했다면 해당 fixture 조건의 검색 동작 증거로만 기록한다.
실제 embedding의 검색 품질이나 다른 corpus로 일반화되는 개선을 검증했다고 표시하지 않는다.

## 검증·배포 범위

검색 증거(source ID 기준 precision/recall·관계 근거 coverage·불필요/숨은 근거 제외)와
실제 모델 답변의 관계 질문 성공·faithfulness·citation 품질을 별도 기록한다.
fake 모델은 orchestration·권한·비용 계약 증거이며 실제 모델의 품질 증거가 아니다.
초기 prototype 기능 판정은 cache-off 검색 비교와 fake 답변/API 회귀, 실제 PG·Neo4j 동작으로 한다.
실제 모델 품질 평가는 승인된 유료 범위가 마련될 때 실행하며, 그 전에는 미평가로 표시한다.
정식 release 품질 gate를 이 작은 fixture 결과로 대체하지 않는다.

latency·비용·projection lag도 함께 측정하되 정량 SLO는 baseline과 함께 고른다.
중간 restricted node/edge, 교차 scope, stale projection, 중복 sync, 재시작, 삭제 수렴,
Neo4j 장애와 generation 도중 relation drift를 검증한다. fake 계약과 실제 DB 증거를 구분한다.
후속 D.1에서 cache invalidation을 추가 검증하며 그것을 E의 선행 조건으로 두지 않는다.
향후 shadow·제한 활성화와 rollback은 graph 선택 flag로 수행한다. 원본/audit는 보존한다.
사용자 화면과 citation shape는 그대로다. Knowledge Map 확대는 별도 제품 요구가 있을 때 진행한다.

## 기존 자산과 library 선택

`backend/app/api/v1/knowledge.py`의 현재 map은 PostgreSQL 조립 결과다. Neo4j 구현 증거가 아니다.
`rag/retrieval.py`, `serving_contracts.py`, `evidence_projection.py`, `indexing.py`,
`serving_generation.py`를 경계로 재사용한다. Neo4j driver는 공식 라이브러리를 사용한다.
`neo4j-graphrag` adapter는 요구에 맞는 검색 기능을 재사용할 때 도입한다. 이를 쓰기 위해
별도 graph/vector/LLM framework를 재작성하거나 PostgreSQL embedding authority를 이중화하지 않는다.
SDK 채택 여부·버전·호출은 E-1 착수 시 lock과 최신 공식 문서로 확인한다.

공식 참고: [Neo4j GraphRAG retriever 가이드](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)
(2026-09-20 조회). 문서의 범용 예시가 ParaWorks 권한·보존 계약을 대체하지 않는다.

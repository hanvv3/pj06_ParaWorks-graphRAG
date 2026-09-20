# D.1 — PostgreSQL 답변 캐시 설계

상태: **C-1 저장소와 C-2 runtime 연결 구현, C-2 독립 리뷰 대기; C-3 미착수**. D 기능 baseline과 E 최소 GraphRAG 뒤 진행한다.
[공통 계약](2026-09-20-remaining-deliverables-design.md) / [계획](../plans/2026-09-20-d1-answer-cache.md).

순서: `D 기능 baseline → E 최소 GraphRAG → D.1 캐시 → Slack → 정식 릴리스 준비`.
E의 cache-off 검색·권한·최종 relation 검증을 재사용한다. D release green은 착수 선행 조건이 아니며,
정식 live 평가와 남은 release gate는 계속 미완료로 추적한다. 제품 변경·유료 실행·flag 활성화는 별개다.

## 첫 버전의 목표

동일한 현재 근거와 보안 문맥에서 답변 생성 호출을 줄인다.
**권장 lookup 위치는 fresh retrieval 이후, generation 이전**이다.
기존 evidence window·전체 input/influence identity를 사용하므로 Assistant 문맥이나
새로 추가된 관련 근거를 무시하는 query-only cache를 만들지 않는다.
pgvector query embedding은 계속 발생할 수 있다. 초기 성과를 embedding 절감으로 보고하지 않는다.

## 저장 계약

- PostgreSQL이 authoritative, default disabled. SQLite는 `NullAnswerCache`.
- 현재 검증을 통과한 substantive answer의 선택 answer blocks만 value로 저장한다.
  실패/안전 canned/no-match 답변은 첫 버전 캐시 대상에서 제외한다.
- value의 유효성 검증에는 모델이 본 **전체 ordered influence set**의 identity/version이 필요하다.
  citation으로 선택된 subset만 검증해서는 안 된다.
- E에서 추가한 전체 path dependency도 포함한다. endpoint 문서가 같아도 node·edge·중간 근거의
  canonical identity/version·승인·scope/permission이 바뀌면 hit가 유효하지 않다.
  E의 내부 provenance carrier를 재사용하며 Neo4j projection generation만 비교하지 않는다.
- exact principal/workspace/security/permission scope와 graph/prompt/model/output/retrieval/
  policy/key 버전, 준비된 전체 답변 입력 HMAC이 key/검증에 포함된다.
- question·prompt·URL·snippet은 cache에 복제하지 않는다. identity/hash/dependency 참조만 둔다.
  answer text 자체는 민감할 수 있으므로 같은 접근·보존 제한을 적용한다.
- 최대 보존은 24시간. 착수 시 이 범위 안에서 TTL을 설정하고 만료·정리 정책을 테스트한다.
  access로 TTL을 연장하지 않는다. 키 교체·근거 철회·권한 변경은 TTL과 무관하게 hit를 무효화한다.

## hit/miss와 비용

hit에서도 현재 권한/근거·전체 영향 집합·E의 relation dependency를 검증하고 citation은
PostgreSQL에서 다시 만든다. 읽기와 최종 노출 사이의 문서 또는 relation revoke/version/permission
변경은 E에서 확장한 finalization 검증으로 차단한다. cache-off와 hit 모두 같은 현재성 기준을 따른다.
cache 저장소만 실패하면 safe miss다. 권한/현재 근거의 authority를 읽지 못하면 fail-closed다.

hit마다 새 AgentRun/Audit와 정확한 비용 기록이 필요하다. query embedding이 실행됐다면 그 실제
비용을 남기고 generation은 미호출/0으로 기록한다. 이전 실행의 비용 receipt나 canned 응답으로
위장하지 않는다. 기존 `rag_finalization.py`의 substantive 경로와 충돌하지 않게 versioned
cache-hit 경로를 정의하는 것이 C-2의 핵심이다. 출처/권한/공개 응답은 그대로 유지한다.

## 검증과 rollout

E에서 먼저 고정한 같은 fixture·principal·retrieval 예산·모델 설정으로 cold/warm generation
호출 수, 실제 비용과 절감량, p50/p95, hit-rate, DB 작업 수를 비교한다. GraphRAG의 검색 개선은
E의 cache-off pgvector/graph 비교로, 캐시 절감은 같은 backend의 cold/warm 비교로 각각 보고한다.
fake 모델의 예상 비용과 실제 provider 비용을 구분하고 미실행 live 품질/비용을 추정값으로 통과시키지 않는다.
성능 목표는 측정 후 정하되 누출/잘못된 citation은 0이어야 한다.
다른 사용자, 같은 role의 다른 사용자, 권한 축소, 선택/비선택 근거 revoke, 새 관련 근거,
relation 변경/철회, 입력 문맥·key/policy 변경, TTL, finalization 중 drift가 모두 miss/거부로 닫혀야 한다.

cache flag off는 E를 포함한 fresh retrieval·generation 경로로 복귀하며 과거 audit를 삭제하지 않는다.
graph flag off는 기존 keyword/pgvector로 복귀하고 cache key/backend version을 분리한다.
Redis L2·semantic similarity cache·embedding reuse·분산 stampede 제어는 초기 범위가 아니다.
동시 miss는 각 요청의 기존 예산 안에서 처리하고 측정된 중복 비용이 있을 때 single-flight를 검토한다.

## 재사용 지점과 열어둔 구현 선택

`backend/app/rag/retrieval.py`, `evidence_projection.py`, `serving_generation.py`,
`backend/app/agent_runtime/rag_finalization.py`, `rag_graph.py`를 재사용한다.
lookup/miss/error/expiry의 작은 port, PostgreSQL store, Null store로 충분하다.
정확한 table/column/API는 C-1에서 확정한다. 권한/보존/비용 계약을 바꾸지 않는 내부 선택은
구현자가 결정한다. RAG state나 기존 extraction cache를 답변 캐시라고 재명명하지 않는다.

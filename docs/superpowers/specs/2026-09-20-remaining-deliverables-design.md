# 남은 Deliverable 공통 설계와 재계획 결정

상태: **비상용 프로토타입 우선 재계획 승인 반영** / 2026-09-20 / 점검 기준 `116347d`.
순서·완료 기준 분리는 승인됐다. 아래 R1의 정식 release 계약 변경은 보류이며 코드에는 적용하지 않았다.
로드맵: [plan.md](../../../plan.md). 각 단계 spec/plan은 이 계약을 참조하고 복사하지 않는다.

## 목적과 적정 규모

3인 개발팀이 근거·권한·비용을 유지하며 비상용 프로토타입의 GraphRAG 효과와 최적화를 먼저 검증한다.
당장 수행할 작업은 파일 책임·실패 조건·완료 증거까지, 먼 단계는 결과와 연결 계약까지 작성한다.
문서 길이의 기준은 줄 수 자체보다 다음 변경을 결정하는 데 필요한 정보다.
계획에는 요구와 검증을, 코드에는 클래스/함수/enum의 정확한 현재 정의를 둔다.

## 승인된 실행 순서와 완료 기준

`D 기능 기준선(F-1/F-2) → E 최소 GraphRAG → D.1 캐시 → Slack → 정식 출시 준비`

- **기능 기준선 통과:** 기존 graph/API/Assistant의 fake 회귀와 실제 PG/pgvector runtime 계약 검증.
  아직 새로 실행한 결과는 없다. 이 결과가 E 착수 조건이며 정식 release green은 아니다.
- **기능 효과 확인:** E는 캐시 없이 관계 검색을 비교하고 D.1은 cold/warm 생성 호출 절감을 따로 비교한다.
  fake 모델 성공을 실제 답변 품질이나 실제 청구 절감의 증거로 표시하지 않는다.
- **정식 release green:** 기존 30-case 승인 실행·reviewer/OAuth·quality publication과 잔여 운영 검증.
  이 작업은 보류하며 D.1/E의 선행 조건에서 제외한다. 미해결 P1을 닫거나 gate를 우회하지 않는다.
- 출시 절차의 reviewer는 지식 승격용 Review Queue와 다르다. 후자의 승인·근거 경계는 지금도 유지한다.
- 프로토타입은 로컬/제한된 검증 환경을 대상으로 한다. 외부 공개·실제 사용자 운영은 이번 승인 범위가 아니다.
  작은 유료 품질 비교도 데이터·모델·호출/비용 상한·실행 범위 확인 뒤 기존 runtime 통제로만 수행한다.
  정식 release 승인 envelope를 임의 발급하지 않으며, runtime readiness/provider latch를 끄지 않는다.

## 계속 유지할 계약

1. PostgreSQL이 권한·현재 버전·승인/철회·audit의 기준이다. pgvector는 기본 벡터 검색이다.
   캐시·Neo4j·LLM 출력은 독립적인 승인 권한이 아니다.
2. 실제 LangChain Runnable, LangGraph StateGraph, 기존 router/registry를 사용한다.
   요청 state와 cross-request cache는 다르다. RAG에 durable checkpointer를 추가하지 않는다.
3. 현재 C.5 정책/Review 승인만 trusted promotion을 만든다. 원본 observation과 pending AI 후보를
   trusted fact로 혼동하지 않는다. 새 AI relation도 같은 원칙을 따른다.
4. 모델 호출 전 권한 필터, 전송 시점과 최종 노출 시점의 현재 근거 검증을 유지한다.
   선택 citation뿐 아니라 답변에 영향을 준 전체 근거를 추적한다. 실패하면 해당 답변을 노출하지 않는다.
5. `/ask`, `/search`, Assistant의 공개 계약과 화면 깊이를 유지한다. citation은 서버가 현재 근거로 만든다.
6. 유료 호출은 승인·예산·durable claim 뒤에만 한다. 실제 usage/unknown reserve를 보존하고
   자동 재호출로 비용 한도를 우회하지 않는다. 키/질문/prompt/provider 원문을 audit에 기록하지 않는다.
7. 기본 disabled, 작은 검증→shadow→제한 활성화. 완료 판정과 운영 rollout 승인은 분리한다.
8. fake 모델·fake connector로 자동 검증한다. SQLite smoke와 실제 PostgreSQL 운영 증거를 구분한다.
9. 기존 개인정보 보존 정책을 유지한다. D.1의 제한된 검증 answer-block 보존은 그 spec에서만 정의한다.

## R1: 실행 권한과 위협 모델(변경안)

**상태: 정식 출시 준비까지 보류.** 이번 순서 변경 승인을 이 계약 변경의 승인으로 확대하지 않는다.

**문제:** 최신 두 리뷰는 모듈 registry 복제와 `used` 초기화로 capability를 다시 소비했다.
이는 재현된 결함이며 NOT CLEAN 판정은 유효하다. closure/비공개 속성으로 이동해도
임의 Python 코드가 실행되는 같은 interpreter를 격리할 수 없다.
같은 DB 자격증명을 그대로 제공한 채 상태를 DB로 옮기는 것도 이 격리를 제공하지 않는다.

**권장:** 배포된 애플리케이션/runner 코드, 운영자, 그 프로세스의 자격증명은 신뢰한다.
API 입력·수집 콘텐츠·모델 응답·외부 제공 식별자는 신뢰하지 않는다.
공식 호출 경계의 변조·권한 누락·재사용, 작업자 간 경쟁, crash/restart는 검증 대상이다.
Python handle은 오사용 방지용이며, durable claim·비용·보고서 발행의 권한은 DB가 갖는다.
권한 검증을 우회한 외부 입력은 계속 거부한다. 런타임 침해는 배포/비밀/OS 격리 문제로 다룬다.

| 대안 | 해결하는 문제 | 판단 |
|---|---|---|
| memory registry를 계속 숨김 | 정상 호출자의 오사용 일부 | 현재 리뷰 루프를 해결하지 못함 |
| 기존 ledger의 claim/CAS/unique report + 신뢰하는 runner | 중복 호출·다중 worker·재시작·중복 발행 | 현재 규모에 권장 |
| 자격증명을 공유하지 않는 별도 authority 서비스/OS 격리 | 비신뢰 실행 코드를 운영해야 하는 경우 | 그런 요구가 생길 때 별도 설계 |

현재 여섯 release table에는 authorization state/owner/fence, dispatch identity,
승인별 unique quality report가 이미 있다. 이를 먼저 사용한다. 새 capability table이나
서비스가 필요하다는 구체적 gap 없이 추가하지 않는다. pure scoring은 반복 계산 가능하게,
외부 호출과 권한 있는 report publication은 단일 실행으로 보장하는 안을 권장한다.

**영향/결정:** 기존 in-process unforgeable capability·계산 단일성 계약의 변경이다.
보류된 D-0에서 명시적으로 확정하고 해당 테스트의 기대를 변경 이유와 함께 이관한다.
현재 코드·리뷰 결과를 CLEAN으로 재명명하거나 안전 계약을 조용히 지우지 않는다.

## R2: 검증과 문서 유지(재계획 운영안)

- 작은 변경: 해당 실패 재현 + 직접 영향 테스트. 상태/보안/비용 경계는 독립 리뷰를 받는다.
- 한 slice 리뷰에서 spec과 코드 품질을 함께 확인하는 것을 기본으로 한다. 병렬 전문 리뷰는
  독립적인 위험이 있을 때만 추가한다. 불명확한 threat model을 테스트 개수로 해결하지 않는다.
- 두 번의 수정 후 같은 원인이 남으면 가정·계약을 재평가한다. 이는 실패를 승인하는 규칙이 아니다.
- 기능 통합 시 관련 provider-free/실제 PostgreSQL/frontend 증거를 확보하고 정식 출시 시 전체 게이트를 실행한다.
  어느 범위를 검증했는지 명시하고 필수 실패/skip을 숨기지 않는다.
- docs-only 변경으로 제품 suite를 모두 반복하지 않는다. 검증 코드 revision과 현재 HEAD의 diff를
  확인해 제품·fixture·dependency가 같음을 기록하고 최종 Git/preview identity는 새로 확인한다.
  기존 authorization/plan HMAC은 재사용하거나 자동 갱신하지 않는다.
- 문서 진입은 roadmap/handoff + 공통 spec + 선택한 단계다. 이력 전문과 완료한 task는 기본 입력에서 뺀다.
- 미래 코드 listing, 후보 schema 전체, 모든 타입 alias를 미리 고정하지 않는다.
  shared/public/storage contract가 바뀌면 해당 결정과 migration 영향을 기록한다.

## 후속 범위

| 대상 | 착수 조건 | 작은 첫 산출물 | 완료/보류 조건 |
|---|---|---|---|
| Slack 복구 | E와 D.1 이후 | 로컬 synthetic fixture로 과거 10개 실패를 재분류 + connector 계약 재현 | 실제 workspace/export/대체 source는 접근·데이터 선택 뒤 연결 |
| Redis L2 | D.1에서 DB/cache 병목 측정 | 지연·hit-rate·쿼리 수 비교 | 근거 없으면 미도입; 권위는 PostgreSQL |
| CDC/streaming | E projection backlog/lag·재조정 비용이 실제 병목 | bounded reconciliation과 동일 corpus 비교 | 성능 필요가 증명되면 별도 전달/복구 계약 작성 |

Slack은 SourceEvent/registry/공유 ingestion 경계를 유지하고 scope/signature/evidence를 보존한다.
실제 데이터가 없다는 이유로 테스트 실패를 제거하지 않는다. synthetic 성공을 live 성공으로 표시하지 않는다.
구체적인 S-1~S-3은 [Slack spec](2026-09-20-slack-recovery-design.md)과
[계획](../plans/2026-09-20-slack-recovery.md)을 따른다. 실제 source 미확보는 다른 기능의 대기 조건이 아니다.
Redis/CDC는 별도 제품 milestone이나 지금 구현할 dependency가 아니다.

## 상세 참조를 읽을 때

기존 [D spec](2026-08-30-deliverable-d-core-rag-answer-graph-v2-design.md)의 §7~16은 구현된 계약,
§13.2는 release 상태/비용, §18은 검증, §19/20은 후속 방향 참조다.
해당 작업이 바꾸는 절만 읽는다. 미승인 변경은 기존 계약을 대체하지 않는다.
이전 foundation §20의 outbox/Knowledge Map 선행 순서는 현재 E 초기 범위에 적용하지 않는다.

기술 근거(2026-09-20 확인): [PostgreSQL locking](https://www.postgresql.org/docs/current/explicit-locking.html)은
트랜잭션 경쟁 제어를, [role/security 문서](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)는
DB 소유자·특권 role의 경계 한계를 설명한다. DB CAS를 프로세스 격리로 해석하지 않는 근거다.

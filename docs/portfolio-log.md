# ParaWorks Portfolio Log

## 2026-09-21 — D.1 C-1 standalone authenticated answer cache

- Added an independent PostgreSQL cache port/model/migration; production answer
  reuse is deliberately left to C-2. The factory is default-off and SQLite is a
  no-I/O Null store. No E/runtime authority, public response, permission, Review
  or cost policy changed. [C-1 consumer contract](superpowers/runbooks/c-1-answer-cache-storage.md).
- Keys bind exact principal/workspace/scope, full prepared input and ordered
  model influences, E graph paths, model/prompt/output/retrieval/graph/backend/
  policy/key identities. Only real-validator-authenticated substantive selected
  blocks and dependency references are stored; no duplicated prompt/question/
  citation metadata or provider receipts. Signed timestamps impose a fixed
  one-hour non-sliding TTL, with a 24-hour DB cap and bounded expired-row cleanup.
- RED/GREEN covered missing storage, new hit identity, migration head alignment,
  and expiry crossing during reads. Actual leased PostgreSQL storage/cleanup,
  signed mutations, scope isolation and full upgrade/downgrade/re-upgrade passed:
  **8 focused tests**, four existing Alembic deprecation warnings. The adjacent
  answer schema/migration/default-runtime/SQLite selectors passed **78 tests,
  6 skipped**, 21 existing Alembic warnings; skips are gated legacy PG suites,
  separate from the actual PG C-1 evidence. Ruff and `git diff --check` passed.
- The broad root `pytest` selector also discovered unrelated scratch code that
  attempts local PostgreSQL at import and failed collection. An all-backend run
  was then stopped on controller direction; neither is claimed as full-suite
  success. Focused selectors above are the completion evidence.
- No paid model/provider calls, `.env` changes, flag activation or push occurred.
  C-2 must still revalidate canonical evidence/relations at publication, rebuild
  citations and account a new run/audit with zero generation cost. Cache savings
  and live quality are not yet measured; formal release NOT CLEAN/P1 remain.

## 2026-09-21 — E-3 fixed-corpus GraphRAG comparison and rollback

- Actual PostgreSQL plus official Neo4j driver ran E-1's same corpus/principal,
  cache-off 50-candidate/5-visible comparison. Relation preserves ordered
  `history_event:1`, `chunk:1` and adds `chunk:2`; evaluation-only public-source
  dedup raises recall 0.5→1.0 while precision remains 1.0. Single/restricted/
  revoke retain their controls; absent returns no evidence. E-3 R1 made the
  module standalone, fed the identical case-specific request to both arms, and
  measured fixed-fixture retrieval/sync: n=5 nearest-rank p50/p95 ms is
  pgvector 59.468/437.342, graph enrichment overhead 5.815/176.653, derived
  seed+overhead total 65.282/613.994, sync 161.109/173.114; lag 0. Ruff had
  removed the first fixture import; explicit fixture alias R2 is now retained.
  The corrected standalone selector: **8 passed in 5.79s**. R2 code revision:
  `d307a2804c3aa9c8f955de386df8dd5ff12ad7d8`.
- Actual default-off composition returns the unwrapped seed. Fake-store unavailable/
  stale tests preserve a controlled existing receipt; they do not claim paid-call
  savings. Existing E-2 actual-driver outage/composition evidence remains separate.
  Controller-coordinated Neo4j restart then fresh official driver/store recovery
  passed (**1 in 25.74s**). A controlled PostgreSQL restart and fresh engine/session
  also reproduced pgvector+graph (**1 in 16.34s**). [E-3 runbook](superpowers/runbooks/e-3-graphrag-comparison.md)
  records commands and limits. Actual-model quality, paid cost, rollout/RBAC,
  formal release NOT CLEAN, and capability P1 remain deferred.

## 2026-09-21 — E-2 permission-preserving relationship retrieval

- E-2 review R1: `/ask`의 기존 8개 seed를 graph cap5로 줄이던 fallback 회귀를
  3개 RED로 재현했다. seed 순서·전체 근거·receipt를 보존하고, graph 공간이 없으면
  조회 전에 enrichment를 생략한다. graph 후보는 `edge_id` 정렬 후 LIMIT을 적용해
  물리 방문 순서가 달라도 동일한 제한 부분집합을 고른다(추가 RED 1개).
  E-2/keyword/pgvector 집중 검증은 실제 PG/Neo4j 설정으로 **117 passed in29.23s**.

- default-off graph enrichment Runnable을 실제 PostgreSQL request/finalizer 조립에
  연결했다. 기존 keyword/pgvector seed와 embedding receipt를 재사용하고, 합계 candidate50 /
  evidence5 예산 안에서 공식 Neo4j driver의 1-hop `SUPPORTED_BY`만 조회한다.
- ordered graph path v1을 prepared influence v3에 바인딩하고 PostgreSQL에서 실제
  provider-send 직전 및 최종 노출 직전 재구성한다. unselected edge hash 변경도 전송 전
  0 call 차단 / 전송 후 1 call 실제 비용 유지 + 답변/citation redaction으로 검증했다.
  공개 응답·Review trust·permission·budget 정책은 유지했다.
- 실제 leased PG + Neo4j 조회·scope/stale fallback·canonical final dependency 검증과,
  실제 LangGraph/production 조립 + fake provider의 전송·최종 결과 검증을 분리했다.
  후자의 PG synchronization은 test port이며 actual PG end-to-end paid dispatch 증거가 아니다.
- actual PG keyword fallback에서 tuple-array 오류를 RED로 확인하고 SQL 실행 bind만
  list로 수정했다. E2+default runtime+기존 graph는 **79 passed in156.29s**;
  최종 영향 suite(11개 파일)는 **318 passed in150.23s**. 상세 명령과 한계는
  [E-2 runbook](superpowers/runbooks/e-2-relationship-retrieval.md) 및 로컬 E-2 report.
  마지막 trace latency가 graph 조회 시간도 포함하도록 RED/GREEN으로 보완한 뒤
  E-2 실제 DB 포함 집중 suite는 **20 passed in26.57s**였다.
- 최초 영향 run의 7개 finalization fixture 실패는 E-1 `350a367`에서도 재현됐다.
  fake authority의 shared-health transport 연결과 real PG schema lease만 보완한
  별도 test-only commit `afda490` 후 **48 passed in3.01s**. 제품 recovery 안전장치는
  바꾸지 않았다. 기존 `rag_finalization.py:125` Ruff SIM117은 미변경 baseline 경고다.
- paid 호출·`.env`·push·rollout·cache 없음. E-3 비교와 독립 리뷰가 다음이며 실제 모델
  품질과 배포 RBAC는 미평가, 정식 release는 **NOT CLEAN/deferred**다.

## 2026-09-21 — E-1 canonical Neo4j projection

- E-1 review R1: scan 진입의 첫 `FOR SHARE`가 timeout 설정보다 앞서고 sweep에는 설정이
  없던 P2를 수정했다. 공통 transaction-local 제한을 첫 corpus 잠금 전에 적용하며
  caller commit/rollback 책임과 예외 전파를 유지한다. 실제 PG의 별도 session이 corpus row를
  `FOR UPDATE`로 잡은 scan/sweep 두 경로가 RED에서 4초 watchdog `57014`로 실패했고,
  수정 후 2초 lock timeout `55P03`로 종료됐다. 기존 순차 generation/cursor fence 검증과
  이번 실제 PG lock contention 검증은 다른 증거다. 집중 projection/baseline/DB suite는
  **18 passed in 12.39s**; 전체 suite 반복은 하지 않았다.

- `349b7cd` 기준에서 최소 `SUPPORTED_BY` projection을 추가했다. 승인/원본 canonical
  resolver와 scope fingerprint를 재사용하며 원문/embedding은 graph에 저장하지 않는다.
  ordered node/edge provenance v1, 제한된 batch/cursor, atomic replay, generation fence,
  삭제/revoke/supersede sweep, lag/count를 제공한다. 검색 adapter/flag/public API는 미변경이다.
- 고정 3-source/1-approval corpus의 **실제 PG cache-off baseline 5 passed**를 구현 전에
  확보했다. 관계 질문은 `history_event:1`, `chunk:1`만 검색해 source 2를 놓쳤다.
  상세 기대 ID/버전/예산/제약은 [E-1 runbook](superpowers/runbooks/e-1-graph-projection.md).
  실제 pgvector baseline에서 tuple의 SQL array 바인딩 실패를 RED로 재현해 세 bind를 list로
  고쳤다. 테스트 vector는 결정적이며 유료 embedding/model 호출은 0이다.
- RED: projection 부재 6 unit + 1 integration 실패, oversized source/policy bound 2 실패,
  incomplete-generation lag 1 실패, driver ValueError 비밀 노출 1 실패를 확인한 뒤 수정했다.
  최종 focused 명령은 `test_graph_projection.py`, `test_graph_projection_baseline.py`,
  `test_graph_projection_neo4j.py`, `test_rag_v2_pgvector_retriever.py`,
  `test_pgvector_store.py`, `test_rag_default_runtime.py` → **85 passed in 36.25s**.
- 실제 PostgreSQL leased schema → Neo4j Community `2026.08.1`, 공식 driver `6.3.1`에서
  atomic rollback/crash, 중복 replay, worker 재시작, scope isolation, revoke/delete,
  source revision supersede, stale generation/cursor 거부를 검증했다. 별도 server restart
  gate도 committed cursor 2와 node 2를 보존한 채 4 nodes/2 edges로 수렴했다:
  **1 passed in 91.81s**(운영자 재시작 대기 포함). 낡은 socket 1회 실패 후 driver 재연결은
  예상된 장애 증거이며 새 provider 호출은 없었다. 각 DB test의 scope/schema cleanup이 수행됐다.
- exact disposable Neo4j 컨테이너를 중지한 상태에서 기존 default runtime의 search/ask/Assistant
  정상·no-match selector를 실행해 **5 passed, 12 deselected in 9.75s**를 확인했고, 같은
  컨테이너를 다시 시작했다. 이는 graph-off 기존 경로의 장애 독립성 증거다.
- fake-driver 테스트는 sanitized failure와 cursor 미전진 계약이고, 위 실제 DB 결과와 구별한다.
  최소권한 운영 RBAC, E-2 path 소비/최종 재검증, E-3 검색 개선과 실제 모델 품질은 미검증이다.
  기존 keyword SQL tuple bind도 E-2 fallback에서 재현할 구체적 점검 항목으로 남긴다.
  `neo4j`/`pytz`만 lock 추가, Ruff/lock consistency/diff checks 통과. paid API, `.env`,
  push/rollout 없음. 정식 release **NOT CLEAN/deferred** 상태는 유지한다.

## 2026-09-20 — F-2 실제 PostgreSQL/pgvector 및 local-fake UI 기준선 검증

- final correction code revision은
  `9200ea3edd02590ef29e9156c3ea7f3a8fc27bf6`이며, F-2 evidence baseline commit은
  `a7a58f8`이다. 이 항목을 기록하는 docs follow-up은 그 code revision 뒤에 있다.
- disposable `paraworks_rag_test` PostgreSQL/pgvector `0.8.2`에서 migration,
  app/vector schema guard와 supplied non-superuser role을 확인했다. live provider,
  `.env`, Docker volume, rollout은 변경하지 않았다.
- F-2 exact real-PG suite는 **318 passed, 13 warnings**, PG selector skip 0이다.
  비용/transaction recovery, C.5/advisory, permission/hidden match·evidence
  revoke, readiness, fake embedding의 native pgvector write/search를 확인했다.
- RED/GREEN으로 b5 check 중복, leased schema `checkfirst` public 오인, paid
  phase-2 safety authority, commit 뒤 expired ORM read의 transaction 재개를
  보완했다. safety transport는 exact shared advisory/runtime health만 수락한다.
- frontend lint/type/build과 controlled-fake Chromium desktop UI evidence가
  통과했다. 48-case run의 두 timing-sensitive initial failures는 fresh retry에서
  passed/no failed tests가 됐다. cleanup은 leased schema 0, active peer session 0.
- 결론: **D functional baseline passed / E-1 may start / formal release deferred
  and NOT CLEAN**. capability P1, OAuth/reviewer, 30-case quality, actual-model
  quality와 live rollout은 통과로 표시하지 않는다.

## 2026-09-20 — F-1 provider-free RAG 기능 기준선 검증

- 최초 provider-free baseline 검증 code revision은
  `5af21b5fc957f5aa2a57dd22097fe73b0aa1d2c5` (`codex/rag-orchestrator-agent`)다.
  그 최초 실행의 evidence-only 기록은 `3495a80`이고, review round 1의 unselected-influence
  regression은 `611c033`에 추가했다. 시작/종료에 확인한 나머지 CRLF/stat-only 문서 `M` 항목은
  기존 변경이며 product diff/staged diff는 없었다; F-1은 production code를 변경하지 않았다.
- 우선 명령을 fake provider/SQLite와 zero-network guard로 실행했다:
  `uv run --no-cache --locked pytest backend/tests/test_rag_v2_graph.py backend/tests/test_rag_default_runtime.py backend/tests/test_rag_api_delivery.py backend/tests/test_rag_v2_provider_free_golden.py backend/tests/test_rag_v2_sqlite_smoke.py -q` →
  **346 passed in 211.95s**, failure/skip 0. 정상 답변·no-match·actor permission denial,
  citation/full-model-influence 변조, 전송 전/생성 후 evidence revoke, cost/audit와 zero external
  provider/network guard를 실제 graph/facade 경로에서 확인했다.
- 직접 영향 명령:
  `uv run --no-cache --locked pytest backend/tests/test_assistant_task18.py backend/tests/test_rag_v2_finalization.py backend/tests/test_rag_v2_costs.py backend/tests/test_rag_v2_provider_safety.py -q` →
  **156 passed, 1 skipped in 45.51s**, failure 0. skip은
  `PARAWORKS_TEST_POSTGRES_URL`이 없는 disposable PostgreSQL projection-owner gate이며 F-2 범위다.
  Assistant의 단일 committed safe message, finalization의 fresh projection/owner fence,
  reserve/actual/unknown 비용 보존·no-redispatch, provider safety sidecar/disabled latch를 확인했다.
- 실제 기능/coverage gap은 발견되지 않아 RED/GREEN 또는 product code 변경은 없었다. API/router의
  formal `live_gate`·provider readiness·permission/cost 제한을 변경하거나 우회하지 않았다.
- 후속 리뷰가 지적한 unselected model influence의 생성 후 revoke 경계는 두 provider-visible slot을
  실제 graph에 넣어 E1만 선택하고 E2를 생성 뒤 `restricted`로 revoke하는 회귀로 보완했다:
  `test_answer_graph_post_generation_unselected_influence_revoke_redacts_full_product` →
  review round 3에서 phase 2 뒤 parent를 `populate_existing` query로 reload해, captured parent
  total과 final persisted total의 exact equality까지 강화했다. component actual 10/5 token usage,
  `charge_basis=actual`, dispatch 1도 유지한다. 전체 answer product/citation/influence를 redaction하고
  한 번만 dispatch하며, exact committed parent/component cost와 full-influence audit HMAC을 보존한다.
  selector는 **1 passed in 4.09s**, 관련 graph/finalization 영향 명령은
  **95 passed, 1 skipped in 101.34s**, failure 0이었다. 기존 implementation은 이 계약을 만족해
  product code 변경은 없었다.
- controller 최종 재검증 대상 revision `17e978193f9ccbf73903aecdfffee8a64561ab0f`에서
  동일 provider-free 우선 명령은 새 회귀를 포함해 **347 passed in 199.09s**, 직접 영향 명령은
  **156 passed, 1 skipped in 46.04s**였다. failure는 0이며 skip은 위와 같은 F-2 전용
  disposable PostgreSQL projection-owner case다.
- 이 기록은 provider-free F-1 증거만 뜻한다. 실제 PostgreSQL+pgvector/UI 통합은 F-2에서,
  actual-model 품질·30-case formal release와 capability P1/NOT CLEAN 해소는 보류된 D-0~D-5에서
  별도로 검증해야 하며, 이번 결과로 이를 통과라고 표시하지 않는다.

## 2026-09-20 — 비상용 프로토타입 우선 순서 승인 반영

- 사용자 결정: 최적화와 GraphRAG 효과 검증을 먼저 진행하며, 기본 안전장치는 유지한다.
- D를 기능 기준선 F-1/F-2와 보류한 정식 release D-0~D-5로 구분했다.
  다음 개발 작업은 기존 graph/API/Assistant의 fake 회귀와 필요한 TDD 보완이다.
- 순서는 `D 기능 기준선 → E → D.1 → Slack → 정식 출시 준비`다.
  runtime이 formal reviewer/runner에 의존하지 않고 D.1도 E의 기술적 선행 조건이 아님을 코드로 확인했다.
- E의 cache-off 관계 검색 비교와 D.1의 cold/warm 생성 비용 비교를 분리했다.
  Slack은 합성 fixture 검증과 실제 source 연동을 구별한 작은 spec/plan을 추가했다.
- 기존 capability P1/NOT CLEAN은 미해결·보류이며 R1 계약 변경은 아직 확정하지 않았다.
  Review Queue·근거·권한·runtime 비용/provider safety는 후속 출시 절차와 함께 미루지 않는다.
- 수정 범위는 문서다. 제품 코드·DB·설정·유료 호출·테스트 실행·원격 push는 하지 않았다.
- 문서 검증: Markdown 14개 범위, 로컬 링크/앵커 48개, D 계획의 테스트 참조 21개
  (기존 20개 + 명시된 신규 1개) 확인 및 `git diff --check` 통과. archive/제품 파일 변경 없음.
  독립 검토에서 활성 계획의 순서·완료 기준·안전 경계 충돌은 발견되지 않았다.

## 2026-09-20 — 남은 단계 문서 재구성

아래는 순서 변경 전 정리 이력이다. 현재 순서와 다음 작업은 위 최신 항목 및 `plan.md`를 따른다.

- 요청: D Core/D.1/E 계획을 프로젝트 규모에 맞추고 반복 입력·과잉 제약·변경 비용을 줄인다.
- 코드 baseline `3f6cb0f`를 읽기 전용 점검하고 D Core, cache/graph, 문서 구조를 병렬 감사했다.
- 현재 로드맵·handoff를 짧게 분리하고 D Core 5개 구현 작업, D.1/E 각 3개 작업으로 정리했다.
- D Core capability의 최신 NOT CLEAN 결과를 반영했다. 앞선 DB 이전 권장만으로는 같은 권한의
  임의 Python 실행을 격리할 수 없다는 한계를 명시했다. 신뢰 모델 변경은 D-0의 미확정 변경안이다.
- D.1은 fresh retrieval 이후 generation 재사용, E는 bounded provenance projection/retrieval을
  초기 범위로 제안했다. 제품 코드·정책 적용·DB·provider 실행·rollout은 하지 않았다.
- 기존 D spec/plan의 경로와 상세 계약을 보존하고 참조 문서로 표시했다.
  긴 실행 이력은 [portfolio archive](superpowers/archive/2026-09-20-portfolio-history.md),
  [handoff archive](superpowers/archive/2026-09-20-session-handoff-history.md),
  [과거 로드맵](superpowers/archive/2026-09-20-product-plan-history.md)에 보존했다.
- 검증 범위는 문서 링크·실존 경로·상태 일관성·diff·제품 파일 미변경 확인이다.
  backend/frontend 테스트를 이번 문서 변경의 fresh GREEN 증거로 재사용하지 않는다.
- 독립 문서 리뷰에서 실행자 session lock 유지, authorization-bootstrap 연결,
  graph seed 선택, 최종 path 재검증의 네 누락을 보완했다.
- 문서 검증: 로컬 링크/앵커 검사, archive 3개 본문 보존 비교, `git diff --check` 통과.
  기본 문맥과 선택한 D spec/plan 전문을 포함한 읽기 분량은 약 2.5만 줄에서 약 740줄로 줄였다.
  이는 줄 수 비교이며 실제 모델 입력 토큰 측정값은 아니다.

## 확인된 이전 코드 상태

| 항목 | 기록 | 해석 |
|---|---|---|
| 코드 | `61eb6d7`, 문서 HEAD `3f6cb0f` | 이번 점검 기준 |
| Task25-B 마지막 회귀 | focused 184, adjacent 361, provider/secret 216 + 1 skip | 과거 기록 |
| 최신 독립 리뷰 | SPEC/CODE NOT CLEAN | memory authority 위조/재사용 미해결 |
| preview | reader unavailable, dispatch 0, authorization false | 실제 runner가 아직 연결되지 않음 |
| PostgreSQL/live release | 잔여 게이트 | D Core 완료로 표시 금지 |

앞으로는 완료한 결과·코드 revision·검증 명령/요약·남은 한계를 한 항목에 기록한다.
중간 진행 로그나 동일 리뷰 내용을 spec/plan/portfolio/handoff 네 곳에 복제하지 않는다.

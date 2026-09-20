# ParaWorks — 현재 인수인계

갱신: 2026-09-21. 단계: **E/D.1 CLEAN; Slack S-1 implemented, independent review pending; formal release NOT CLEAN**.

## Slack S-1 → S-2

- Code `1ddfc21`, [S-1 runbook](s-1-slack-synthetic-ingestion.md): bounded known-thread
  cursors, raw evidence/participants, restricted context. Fresh tests 93 passed,
  external attempts 0. Historical ten still fail with explicit classification.
- S1 review 뒤 S2를 진행한다. 사용자 승인은 NEW synthetic source의 서버 버전/서명
  검증 확장뿐이며 기존 unsigned 행 자동 신뢰·Review 생략·live 연결 승인은 없다.
- `SyntheticSlackClient`/`SyntheticSlackConnector`는 fixture이며 라벨은 authority가
  아니다. 공유 sync는 Review 0개인 legacy observation만 보존한다. S2는 현재
  source/Review 경계로 pending을 만들어야 한다. 미발견 old thread/관찰 50개 밖의
  thread는 발견 보장이 없다. SourceEvent DTO와 legacy IDs는 유지했다.

## C-3 측정과 다음 단계

- 같은 실제 PG/backend에서 baseline과 cold/warm 생성 호출·DB 작업·시간을 측정했다.
  [C-3 runbook](c-3-answer-cache-comparison.md)에 원시 표·측정 범위·재현 명령·한계가 있다.
  fake 모델 비용이며 embedding 절감이나 live 응답속도 개선을 주장하지 않는다.
- cache-off는 fresh graph generation, graph-off는 별도 key의 keyword로 복귀한다.
  기존 감사 기록을 보존한다. 기본 flag·권한·Review·비용 정책은 바꾸지 않았다.
- C-3 독립 리뷰는 CLEAN이며 Slack S-1 합성 ingestion을 구현했다. 실제 Slack/OAuth·유료 호출·
  rollout은 실행하지 않았다. capability P1과 formal release NOT CLEAN은 유지한다.

## C-2 runtime 연결

- default-off `rag_answer_cache_enabled`를 fresh retrieval/preparation 뒤 연결했다.
  `answer-cache-hit:v1`은 새 run/감사와 generation 미호출·0원, 실제 embedding 비용을 보존한다.
- lookup은 노출 권한이 아니다. 현재 key/value/expiry와 durable parent를 재인증한 뒤 기존
  PG C.5/owner/canonical transaction에서 전체 influence/path와 citation을 검증한다.
- [C-2 runbook](c-2-answer-cache-runtime.md)에 내부 HMAC 소비자, bounded cleanup/운영 명령,
  실제 PG와 fake 테스트 경계가 있다. PG 조립 검증은 E metadata/scorer fixture이며
  전체 migration trigger 또는 정식 release 통과를 뜻하지 않는다.
- 새 의존성/DB migration/공개 DTO 변경은 없다. C-2 R1 독립 리뷰는 CLEAN이다.

## C-1 독립 저장소

- `rag_answer_cache_entries`는 한 시간 고정 TTL과 signed value를 사용한다. factory 기본 off,
  SQLite Null이다. 독립 저장소에 C-2가 graph/finalization 연결을 추가했다.
- E prepared influence v3와 ordered path v1 전체를 인증하고 exact principal/scope·입력·
  model/prompt/output/retrieval/graph/policy/key 버전을 key에 묶는다. 검증된 selected blocks만
  저장한다. [C-1 runbook](c-1-answer-cache-storage.md)의 consumer 계약을 먼저 읽는다.
- C-2는 새 run/audit의 zero-generation 회계, 현재 citation 재구성, hit 이후 최종
  전체 근거/관계 재검증을 연결했다. cache hit은 권한 authority가 아니다.

## E-3 완료 후속

- E-3 fixed corpus comparison shows relation public-source recall 0.5→1.0 at
  precision 1.0; single/absent/restricted/revoke preserve E-1 expectations.
  Default-off/unavailable/stale rollback and receipt reuse are covered. See
  [E-3 runbook](e-3-graphrag-comparison.md).
- E-3 R1 fixed the standalone fixture import and records one raw timing sample
  per fixed case; those measurements are fixture observability, not a latency SLO.
- After controller-coordinated Neo4j restart and PostgreSQL restart/fresh-session
  reconstruction, E-2 retrieval recovered the same relation evidence. No paid
  provider/cache/flag activation occurred.
- E 전체 리뷰는 CLEAN이며 C-1 저장소가 E-2 graph dependencies를 재사용한다.
  Actual-model quality, production RBAC, capability P1 and formal release remain deferred.

## E-2 후속 진입

- default-off `rag_graph_enrichment_enabled`로 기존 keyword/pgvector seed Runnable을
  감싼다. paid embedding/receipt와 50 candidate·5 evidence 예산을 그대로 재사용한다.
- E-2 R1은 기존 `/ask` 8개 seed를 보존한다. graph enrichment 자체는 cap5이며
  seed가 그 공간을 채우면 graph I/O 없이 원래 근거를 반환한다. edge ID 정렬 후 LIMIT으로
  제한 후보를 안정적으로 선택한다.
- ordered graph path v1을 retrieval→prepared influence v3→실제 provider-send C.5 검증→
  최종 PG 검증으로 운반한다. edge만 바뀌어도 전송 전 0 call 차단 또는 전송 후 답변/citation
  redaction으로 처리하며 실제 비용을 보존한다. 공개 응답/Review/권한 정책은 미변경이다.
- [E-2 runbook](e-2-relationship-retrieval.md)에 config, 내부 fingerprint 소비자와 fake/real DB
  검증 경계를 정리했다. keyword tuple-array 오류도 실제 PG에서 재현 후 SQL 경계만 수정했다.
- E-2 독립 리뷰와 E-3 동일 corpus 비교는 완료됐다. 자동 flag 활성화·유료 호출은 하지 않았다.

## E-1 기반 계약

- projection 코드/고정 fixture/실제 DB 검증은 [E-1 runbook](e-1-graph-projection.md),
  fresh 수치는 [portfolio](../../portfolio-log.md)의 최신 E-1 항목에 있다.
- E-2는 `GraphPathDependency` v1 ordered node/edge와 canonical references를
  RetrievalResult→graph state→모델 전송 전→최종 노출 PG 검증으로 운반한다.
  Neo4j scope는 principal까지 포함한 기존 scope fingerprint이며 complete/current generation만
  소비할 수 있다. endpoint-only 검증으로 edge 변경을 통과시키지 않는다.
- Public API/Review/permission/cost 정책은 바뀌지 않았다. 실제 모델 품질·E-3
  검색 효과·배포 최소권한 RBAC는 미검증이다. 공식 Neo4j driver 6.3.1이 lock에 추가됐다.
- E-1 actual PG baseline의 배열 tuple 오류는 pgvector SQL 경계 3곳을 수정했고,
  E-2에서는 keyword `_postgresql_search`의 같은 오류를 실제 PG로 재현·수정했다.
  유료 호출이나 fallback 안전장치를 우회하지 않는다.
- 테스트는 fresh `.tmp` basetemp/cache와 `.venv-task4-r3-review/Scripts/python.exe`를 사용한다.
  기존 `.venv` launcher는 깨져 있어 uv는 `UV_PROJECT_ENVIRONMENT=.venv-task4-r3-review`와
  workspace cache를 명시한다. DB credentials는 프로세스 환경으로만 공급하고 저장하지 않는다.

## 작업 위치

- Worktree: `C:/Users/hanvv/Study/potenup3/pj06_ParaWorks+graphRAG/.worktrees/review-hitl-v2-design`
- Branch: `codex/rag-orchestrator-agent`
- C-1 착수 code HEAD: `c9c94e0bb1ef4445ae2b37eec3df6ad6b515f6b4`
  (E 전체 리뷰 CLEAN). 재개 시 `git status --short`와 HEAD를 다시 확인.
- 진입 문서: [plan.md](../../../plan.md), [공통 spec](../specs/2026-09-20-remaining-deliverables-design.md),
  [D 마무리 spec](../specs/2026-09-20-d-core-completion-design.md), [D 계획](../plans/2026-09-20-d-core-completion.md).

## 현재 사실

- D의 retriever/answer graph·API/Assistant·비용/권한 기반 코드가 존재한다.
  완료된 Tasks1~22를 다시 시작하지 않는다. F-1/F-2 기능/PG 기준선은 검증됐다.
- Task23/24 release ledger·preview/authorization 기반, Task25-A evaluator,
  Task25-B memory capability가 구현돼 있다. D 전체 릴리스 완료를 뜻하지 않는다.
- `3f6cb0f`의 최신 독립 spec/code 리뷰는 **NOT CLEAN**이다. 메모리 registry 복제와
  사용 표시 초기화로 capability를 재사용할 수 있다는 동일 원인의 P1이 남았다.
- 이전 Git source 검증·DTO-only 경로·소비 시 freshness 관련 재현은 수정 기록이 있다.
- CLI의 마지막 기록은 `preview_snapshot_reader_unavailable`, dispatch 0, authorization false.
  실제 reader/30-case runner/reviewer session/quality publication/`run` CLI가 남아 있다.
- F-1/F-2 기능 기준선 증거가 있다. F-2 exact real-PG suite는 318 passed, 13 warnings,
  frontend local-fake gates도 passed다. 상세 selector/cleanup은 portfolio 최신 항목과
  `.superpowers/sdd/d-core-completion/task-F-2-report.md`를 따른다.
- v1 provider rebind 이력은 이전 정책 재료 부족으로 fail-closed다. 정상 recovery/재bootstrap
  결정 없이 실행을 열지 않는다. 대상 데이터가 없는 disposable fixture와 운영 복구는 구분한다.

## 승인된 변경과 다음 작업

2026-09-20 사용자는 **D 기능 기준선 → E → D.1 → Slack → 정식 출시 준비** 순서를 승인했다.
이후 구현 지시로 E와 C-1을 진행했다.
release reviewer/OAuth·30-case runner·quality publication은 보류되며 지식용 Review Queue는 유지한다.
R1 threat-model 변경과 P1 해결을 승인된 것으로 해석하지 않는다. 보류된 D-0에서 결정한다.

1. **E-1~E-3:** 캐시 없이 관계 검색/근거 경로를 구현·비교한다. 이어 **C-1~C-3**에서 캐시를 연결한다.
2. **S-1~S-3:** 합성 Slack의 수집→Review→검색을 재현한다. 실제 데이터 선택은 live 연동 때 묻는다.

계획 승인 이후 E와 D.1 리뷰를 완료했다. 다음 기능은 S-1 독립 리뷰 후 Slack S-2다.

## 검증과 운영 경계

- 과거 `184 passed`, 인접 `361 passed`, provider/secret `216 passed, 1 skipped`는
  `61eb6d7`/`3f6cb0f`에서 보고된 이력이다. **이번 세션의 실행 결과가 아니다.**
- 수정한 경계의 RED/GREEN과 영향 회귀를 먼저 돌리고, 전체 회귀는 통합 지점에서 실행한다.
- no-network fake provider가 기본. 프로토타입 유료 비교는 데이터·모델·호출/비용 상한·실행 범위를
  확인한 뒤 기존 runtime 예산/provider safety로 수행한다. 이번 계획 승인은 유료 실행 승인이 아니다.
- 보류된 정식 30-case gate는 exact clean commit/fixture/DB/authority preview와 별도 승인 후
  최대 USD `0.360000`이다. 이 formal budget을 일반 프로토타입 승인으로 재사용하지 않는다.
- 문서 변경도 Git HEAD를 바꾼다. 기존 preview/authorization이나
  `implementation_plan_reference_hmac`를 자동 재사용/재발급하지 않는다.
- Redis L2·CDC는 측정된 병목이 생길 때 검토한다. 실제 Slack source 미확보는 다른 기능의 대기 조건이 아니다.

## 필요한 기록만 찾아보기

- 과거 세션: [handoff archive](../archive/2026-09-20-session-handoff-history.md)
- 과거 검증: [portfolio archive](../archive/2026-09-20-portfolio-history.md)
- 최신 P1의 공유 가능한 요약: [공통 spec의 결정 R1](../specs/2026-09-20-remaining-deliverables-design.md#r1-실행-권한과-위협-모델변경안)
- 로컬 `.tmp/task25-b-authority-code-r1-review.md`와
  `.tmp/task25b-authority-spec-r1-independent-review.md`는 선택적 재현 자료다.
  ignored 파일이 없는 새 체크아웃에서도 위 요약으로 상태를 파악할 수 있다.
  formal release 재개 때 D-0/D-1 acceptance를 확인한다. 현재 다음 작업은 위 Slack S-1 순서를 따른다.

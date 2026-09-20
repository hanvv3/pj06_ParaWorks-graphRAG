# ParaWorks — 현재 인수인계

갱신: 2026-09-21. 단계: **E-2 relationship retrieval implemented; independent review/E-3 next; formal release deferred and NOT CLEAN**.

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
- 다음은 독립 리뷰와 E-3 동일 corpus 비교다. 자동 flag 활성화·유료 호출·cache는 하지 않았다.

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
- 점검한 code HEAD: `9200ea3edd02590ef29e9156c3ea7f3a8fc27bf6`
  (F-2 evidence baseline `a7a58f8`). 재개 시 `git status --short`와 HEAD를 다시 확인.
- 진입 문서: [plan.md](../../../plan.md), [공통 spec](../specs/2026-09-20-remaining-deliverables-design.md),
  [D 마무리 spec](../specs/2026-09-20-d-core-completion-design.md), [D 계획](../plans/2026-09-20-d-core-completion.md).

## 현재 사실

- D의 retriever/answer graph·API/Assistant·비용/권한 기반 코드가 존재한다.
  완료된 Tasks1~22를 다시 시작하지 않는다. F-1/F-2에서 실제 기능/PG 기준선을 새로 검증한다.
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

사용자가 승인한 것은 **D 기능 기준선 → E → D.1 → Slack → 정식 출시 준비**의 문서 변경이다.
release reviewer/OAuth·30-case runner·quality publication은 보류되며 지식용 Review Queue는 유지한다.
R1 threat-model 변경과 P1 해결을 승인된 것으로 해석하지 않는다. 보류된 D-0에서 결정한다.

1. **E-1~E-3:** 캐시 없이 관계 검색/근거 경로를 구현·비교한다. 이어 **C-1~C-3**에서 캐시를 연결한다.
2. **S-1~S-3:** 합성 Slack의 수집→Review→검색을 재현한다. 실제 데이터 선택은 live 연동 때 묻는다.

이번 턴은 planning 문서 변경까지만 수행했다. 구현 재개 시 기존 subagent-driven 방식을 유지한다.

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
  formal release 재개 때 D-0/D-1 acceptance를 확인한다. 지금의 다음 작업은 E-1이다.

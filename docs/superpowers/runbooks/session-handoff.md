# ParaWorks — 현재 인수인계

갱신: 2026-09-20. 단계: **planning 재정리 완료; 제품 구현 변경 없음**.

## 작업 위치

- Worktree: `C:/Users/hanvv/Study/potenup3/pj06_ParaWorks+graphRAG/.worktrees/review-hitl-v2-design`
- Branch: `codex/rag-orchestrator-agent`
- 점검한 코드 HEAD: `3f6cb0f` (구현 `61eb6d7`). 재개 시 `git status --short`와 HEAD를 다시 확인.
- 진입 문서: [plan.md](../../../plan.md), [공통 spec](../specs/2026-09-20-remaining-deliverables-design.md),
  [D 마무리 spec](../specs/2026-09-20-d-core-completion-design.md), [D 계획](../plans/2026-09-20-d-core-completion.md).

## 현재 사실

- D의 retriever/answer graph·API/Assistant·비용/권한 기반 코드가 존재한다.
  완료된 Tasks1~22를 다시 시작하지 않는다. PostgreSQL 등 잔여 증거는 D-5에서 확인한다.
- Task23/24 release ledger·preview/authorization 기반, Task25-A evaluator,
  Task25-B memory capability가 구현돼 있다. D 전체 릴리스 완료를 뜻하지 않는다.
- `3f6cb0f`의 최신 독립 spec/code 리뷰는 **NOT CLEAN**이다. 메모리 registry 복제와
  사용 표시 초기화로 capability를 재사용할 수 있다는 동일 원인의 P1이 남았다.
- 이전 Git source 검증·DTO-only 경로·소비 시 freshness 관련 재현은 수정 기록이 있다.
- CLI의 마지막 기록은 `preview_snapshot_reader_unavailable`, dispatch 0, authorization false.
  실제 reader/30-case runner/reviewer session/quality publication/`run` CLI가 남아 있다.
- PostgreSQL/live gate의 최신 실행 증거가 없다. 이번 문서 작업에서는 DB 상태나 테스트를 실행하지 않았다.
- v1 provider rebind 이력은 이전 정책 재료 부족으로 fail-closed다. 정상 recovery/재bootstrap
  결정 없이 실행을 열지 않는다. 대상 데이터가 없는 disposable fixture와 운영 복구는 구분한다.

## 다음 작업과 결정

1. **D-0 (planning):** 동일 interpreter의 임의 실행을 경계 밖으로 두는 권장 모델과,
   기존 release ledger를 이용한 durable claim/단일 report publication을 확정한다.
   계획 검토 요청을 이 보안 계약 변경의 자동 승인으로 해석하지 않는다.
2. **D-1 (구현):** 확정된 계약을 회귀 테스트로 고정하고 기존 ownership/state/CAS/unique report를
   연결한다. 숨긴 Python 객체나 새 DB table만으로 런타임 침해를 막는다고 주장하지 않는다.
3. D-2 reader → D-3 runner → D-4 review/publication → D-5 통합 증거 순으로 진행한다.

## 검증과 운영 경계

- 과거 `184 passed`, 인접 `361 passed`, provider/secret `216 passed, 1 skipped`는
  `61eb6d7`/`3f6cb0f`에서 보고된 이력이다. **이번 세션의 실행 결과가 아니다.**
- 수정한 경계의 RED/GREEN과 영향 회귀를 먼저 돌리고, 전체 회귀는 통합 지점에서 실행한다.
- no-network fake provider가 기본. 실제 유료 gate는 exact clean commit/fixture/DB/authority
  preview와 별도 실행 승인 후 최대 USD `0.360000`; 재시도·확대·rollout은 포함되지 않는다.
- 문서 변경도 Git HEAD를 바꾼다. 기존 preview/authorization이나
  `implementation_plan_reference_hmac`를 자동 재사용/재발급하지 않는다.
- D Core → D.1 → E → Slack 순서. Redis L2·CDC는 측정된 병목이 생길 때 검토한다.

## 필요한 기록만 찾아보기

- 과거 세션: [handoff archive](../archive/2026-09-20-session-handoff-history.md)
- 과거 검증: [portfolio archive](../archive/2026-09-20-portfolio-history.md)
- 최신 P1의 공유 가능한 요약: [공통 spec의 결정 R1](../specs/2026-09-20-remaining-deliverables-design.md#r1-실행-권한과-위협-모델변경안)
- 로컬 `.tmp/task25-b-authority-code-r1-review.md`와
  `.tmp/task25b-authority-spec-r1-independent-review.md`는 선택적 재현 자료다.
  ignored 파일이 없는 새 체크아웃에서도 위 요약과 D-1 acceptance로 재개할 수 있어야 한다.

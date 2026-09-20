# ParaWorks — 현재 로드맵

갱신: 2026-09-20. 코드 점검 기준: `3f6cb0f` / `codex/rag-orchestrator-agent`.
이 파일은 제품 방향·문서 탐색·작업 순서의 source of truth다.

## 제품과 실행 순서

한국어 중심 회사 기억 플랫폼. 실제 LangChain/LangGraph로 승인 지식과 원본 근거를
권한에 맞게 검색하고, 같은 Assistant 화면에서 근거 있는 답변을 제공한다.
3인 개발팀의 agent ownership·작은 shared contracts는 [AGENTS.md](AGENTS.md)를 따른다.

`D Core 완료 → D.1 답변 캐시 → E Neo4j GraphRAG → Slack 복구`

이번 요청으로 D.1/E의 방향 설계와 단계별 계획까지 미리 정리한다.
구현 착수 순서와 각각의 독립 완료 게이트는 유지한다. 실제 유료 실행·rollout·push 승인은 별개다.

## 현재 상태와 문서

| 단계 | 현재 상태 | 설계 | 구현 계획 |
|---|---|---|---|
| A/B/C/C.5 | 과거 구현·검증 기록 있음. C.5 rollout disabled | [이력](docs/superpowers/archive/2026-09-20-product-plan-history.md) | 재구현하지 않음 |
| D Core | Task25-B 부분 구현, 최신 리뷰 NOT CLEAN; 릴리스 미완료 | [마무리 spec](docs/superpowers/specs/2026-09-20-d-core-completion-design.md) | [D-0~D-5](docs/superpowers/plans/2026-09-20-d-core-completion.md) |
| D.1 | 계획안, 미구현. D Core 완료 뒤 진입 | [캐시 spec](docs/superpowers/specs/2026-09-20-d1-answer-cache-design.md) | [C-1~C-3](docs/superpowers/plans/2026-09-20-d1-answer-cache.md) |
| E | 계획안, 미구현. D.1 완료 뒤 진입 | [GraphRAG spec](docs/superpowers/specs/2026-09-20-e-graphrag-design.md) | [E-1~E-3](docs/superpowers/plans/2026-09-20-e-graphrag.md) |
| Slack | 마지막 단계; 데이터 소스 선택 필요 | [공통 spec §후속 범위](docs/superpowers/specs/2026-09-20-remaining-deliverables-design.md#후속-범위) | 기존 10개 실패를 보이게 유지 |

## 재개할 때 읽는 순서

1. 이 파일과 [현재 handoff](docs/superpowers/runbooks/session-handoff.md).
2. [공통 계약·재계획 결정](docs/superpowers/specs/2026-09-20-remaining-deliverables-design.md).
3. 선택한 단계의 spec과 **다음 작업 한 개**. [portfolio](docs/portfolio-log.md)의 최신 항목만 확인.
4. 그 작업의 코드·테스트. 기존 긴 문서는 해당 계약이 필요할 때 지정된 절만 조회.

당장 다음 작업은 **D-0: 권장 신뢰 모델·durable 단일 실행 계약 확정(planning)**이다.
이 결정 뒤 D-1부터 실제 구현으로 이어간다. DB 테이블/마이그레이션을 먼저 추가하지 않는다.

## 계획을 유지하는 방법

- 공통 불변조건은 공통 spec 한 곳에, 단계별 결과·검증은 해당 spec/plan에 둔다.
- 파일/클래스 전체 구현을 계획에 복사하지 않는다. 기존 코드가 내부 API의 기준이다.
- 당장 한 작업만 상세화하고 후속 단계의 스키마·SDK 버전·성능 목표는 착수 시 증거로 고른다.
- 일반 구현 선택은 개발자가 해결한다. 권한·승인·비용·보존·공개 계약 변경만 명시적 결정으로 남긴다.
- 리뷰 지적은 재현 가능한 제품/운영 영향으로 분류한다. 같은 원인에 대한 반복 리뷰는
  초점을 좁혀 재검증하고, 두 번 고쳐도 같은 문제가 남으면 가정을 다시 검토한다.
- 문서만 변경했으면 링크·범위·일관성을 확인한다. 제품 테스트를 새로 통과했다고 쓰지 않는다.

## 기존 문서와 우선순위

현재 상태·다음 작업은 이 파일과 handoff가 우선한다. **미승인 계약 변경안은 기존 승인 계약을
자동 대체하지 않는다.** D-0 결정 전 capability P1과 D Core NOT CLEAN 상태는 유지된다.
기존 [D 상세 spec](docs/superpowers/specs/2026-08-30-deliverable-d-core-rag-answer-graph-v2-design.md)과
[27-task 계획](docs/superpowers/plans/2026-08-31-deliverable-d-core-rag-answer-graph-v2.md)은
구현된 wire/storage/cost 계약의 상세 참조로 보존한다. 오래된 체크박스로 진행률을 판단하지 않는다.

원래 루트 `main`과 이 작업트리는 별도 체크아웃이다. 최신 작업 위치는
`.worktrees/review-hitl-v2-design`이며 이번 정리는 main 동기화나 원격 반영을 수행하지 않는다.
